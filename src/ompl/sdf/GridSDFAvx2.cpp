#include "ompl/sdf/GridSDF.h"

#include <immintrin.h>

namespace
{
    /// out[lane] = a[lane] + t[lane] * (b[lane] - a[lane]), lane-for-lane identical to
    /// GridSDF::lerp() -- one sub, one mul, one add, never fused into an FMA (this file
    /// is built with -mavx2 only, no -mfma), so each lane rounds exactly like the scalar
    /// path and distanceBatch/valueGradientBatch stay bit-identical whether or not this
    /// kernel is enabled.
    inline __m256d lerpAvx(__m256d a, __m256d b, __m256d t)
    {
        return _mm256_add_pd(a, _mm256_mul_pd(t, _mm256_sub_pd(b, a)));
    }

    struct CornersAvx
    {
        __m256d v000, v100, v010, v110, v001, v101, v011, v111;
    };

    /// Reads the 8 corner values for 4 cells and packs them into 8 vectors, one per
    /// corner offset. Deliberately *not* `_mm256_i64gather_pd`: measured on this call
    /// site's real batch size (~40 points/call), the gather instruction was ~1.5x
    /// slower than these plain scalar loads -- out-of-order execution already overlaps
    /// the 8 scattered scalar reads well, and the gather's own overhead doesn't pay for
    /// itself at this width. See results/mbm_buffer5mm profiling notes for the A/B.
    inline CornersAvx gatherCorners(const double *values, const long long *base, std::size_t strideY,
                                    std::size_t strideZ)
    {
        const auto at = [&](std::size_t offset)
        {
            return _mm256_set_pd(values[base[3] + offset], values[base[2] + offset], values[base[1] + offset],
                                 values[base[0] + offset]);
        };

        CornersAvx c;
        c.v000 = at(0);
        c.v100 = at(1);
        c.v010 = at(strideY);
        c.v110 = at(strideY + 1);
        c.v001 = at(strideZ);
        c.v101 = at(strideZ + 1);
        c.v011 = at(strideZ + strideY);
        c.v111 = at(strideZ + strideY + 1);
        return c;
    }
}  // namespace

namespace ompl::sdf
{
    void GridSDF::distanceBatchAvx2(const Eigen::Ref<const Eigen::Matrix3Xd> &points,
                                    Eigen::Ref<Eigen::VectorXd> distances) const
    {
        const Eigen::Index count = points.cols();
        const Eigen::Index vecEnd = count - (count % 4);

        alignas(32) long long baseLanes[4];
        alignas(32) double fxLanes[4];
        alignas(32) double fyLanes[4];
        alignas(32) double fzLanes[4];

        for (Eigen::Index i = 0; i < vecEnd; i += 4)
        {
            for (int lane = 0; lane < 4; ++lane)
            {
                const Cell cell = locate<false>(points.col(i + lane));
                baseLanes[lane] = static_cast<long long>(cell.base);
                fxLanes[lane] = cell.fraction[0];
                fyLanes[lane] = cell.fraction[1];
                fzLanes[lane] = cell.fraction[2];
            }

            const __m256d fx = _mm256_load_pd(fxLanes);
            const __m256d fy = _mm256_load_pd(fyLanes);
            const __m256d fz = _mm256_load_pd(fzLanes);

            const CornersAvx v = gatherCorners(values_.data(), baseLanes, strideY_, strideZ_);

            const __m256d x00 = lerpAvx(v.v000, v.v100, fx);
            const __m256d x10 = lerpAvx(v.v010, v.v110, fx);
            const __m256d x01 = lerpAvx(v.v001, v.v101, fx);
            const __m256d x11 = lerpAvx(v.v011, v.v111, fx);
            const __m256d y0 = lerpAvx(x00, x10, fy);
            const __m256d y1 = lerpAvx(x01, x11, fy);
            const __m256d result = lerpAvx(y0, y1, fz);

            _mm256_storeu_pd(distances.data() + i, result);
        }

        for (Eigen::Index i = vecEnd; i < count; ++i)
            distances[i] = interpolateValue(points.col(i));
    }

    void GridSDF::valueGradientBatchAvx2(const Eigen::Ref<const Eigen::Matrix3Xd> &points,
                                         Eigen::Ref<Eigen::VectorXd> distances,
                                         Eigen::Ref<Eigen::Matrix3Xd> gradients) const
    {
        const Eigen::Index count = points.cols();
        const Eigen::Index vecEnd = count - (count % 4);

        alignas(32) long long baseLanes[4];
        alignas(32) double fxLanes[4];
        alignas(32) double fyLanes[4];
        alignas(32) double fzLanes[4];
        alignas(32) long long maskXLanes[4];
        alignas(32) long long maskYLanes[4];
        alignas(32) long long maskZLanes[4];

        for (Eigen::Index i = 0; i < vecEnd; i += 4)
        {
            for (int lane = 0; lane < 4; ++lane)
            {
                const Cell cell = locate<true>(points.col(i + lane));
                baseLanes[lane] = static_cast<long long>(cell.base);
                fxLanes[lane] = cell.fraction[0];
                fyLanes[lane] = cell.fraction[1];
                fzLanes[lane] = cell.fraction[2];
                // All-ones/all-zeros so the mask can be reinterpreted straight into the
                // sign-bit predicate _mm256_blendv_pd reads.
                maskXLanes[lane] = cell.derivativeActive[0] ? -1LL : 0LL;
                maskYLanes[lane] = cell.derivativeActive[1] ? -1LL : 0LL;
                maskZLanes[lane] = cell.derivativeActive[2] ? -1LL : 0LL;
            }

            const __m256d fx = _mm256_load_pd(fxLanes);
            const __m256d fy = _mm256_load_pd(fyLanes);
            const __m256d fz = _mm256_load_pd(fzLanes);

            const CornersAvx v = gatherCorners(values_.data(), baseLanes, strideY_, strideZ_);

            const __m256d x00 = lerpAvx(v.v000, v.v100, fx);
            const __m256d x10 = lerpAvx(v.v010, v.v110, fx);
            const __m256d x01 = lerpAvx(v.v001, v.v101, fx);
            const __m256d x11 = lerpAvx(v.v011, v.v111, fx);
            const __m256d y0 = lerpAvx(x00, x10, fy);
            const __m256d y1 = lerpAvx(x01, x11, fy);
            _mm256_storeu_pd(distances.data() + i, lerpAvx(y0, y1, fz));

            // Same three gradient formulas as GridSDF::interpolate(), computed
            // unconditionally (cheap) and then masked with blendv instead of the
            // scalar path's per-axis `if (cell.derivativeActive[d])` branch.
            const __m256d zero = _mm256_setzero_pd();

            const __m256d dx0 = lerpAvx(_mm256_sub_pd(v.v100, v.v000), _mm256_sub_pd(v.v110, v.v010), fy);
            const __m256d dx1 = lerpAvx(_mm256_sub_pd(v.v101, v.v001), _mm256_sub_pd(v.v111, v.v011), fy);
            const __m256d maskX = _mm256_castsi256_pd(_mm256_load_si256(reinterpret_cast<const __m256i *>(maskXLanes)));
            const __m256d gradX = _mm256_blendv_pd(
                zero, _mm256_mul_pd(lerpAvx(dx0, dx1, fz), _mm256_set1_pd(inverseSpacing_[0])), maskX);

            const __m256d maskY = _mm256_castsi256_pd(_mm256_load_si256(reinterpret_cast<const __m256i *>(maskYLanes)));
            const __m256d gradY = _mm256_blendv_pd(
                zero,
                _mm256_mul_pd(lerpAvx(_mm256_sub_pd(x10, x00), _mm256_sub_pd(x11, x01), fz),
                              _mm256_set1_pd(inverseSpacing_[1])),
                maskY);

            const __m256d maskZ = _mm256_castsi256_pd(_mm256_load_si256(reinterpret_cast<const __m256i *>(maskZLanes)));
            const __m256d gradZ = _mm256_blendv_pd(
                zero, _mm256_mul_pd(_mm256_sub_pd(y1, y0), _mm256_set1_pd(inverseSpacing_[2])), maskZ);

            alignas(32) double gx[4];
            alignas(32) double gy[4];
            alignas(32) double gz[4];
            _mm256_store_pd(gx, gradX);
            _mm256_store_pd(gy, gradY);
            _mm256_store_pd(gz, gradZ);
            for (int lane = 0; lane < 4; ++lane)
            {
                gradients(0, i + lane) = gx[lane];
                gradients(1, i + lane) = gy[lane];
                gradients(2, i + lane) = gz[lane];
            }
        }

        for (Eigen::Index i = vecEnd; i < count; ++i)
        {
            const ValueGradient vg = interpolate(points.col(i));
            distances[i] = vg.value;
            gradients.col(i) = vg.gradient;
        }
    }
}  // namespace ompl::sdf
