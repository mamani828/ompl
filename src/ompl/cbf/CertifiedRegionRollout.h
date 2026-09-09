#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/util/Exception.h>

namespace ompl::cbf
{
    /// Straight-line steering certified in bubbles, as a drop-in `RolloutPlanner`.
    ///
    /// ### What is being reformulated
    ///
    /// A CBF rollout asks the barrier a question about a *control*: given `u`, is it
    /// admissible, and if not what is the nearest admissible one. Answering costs a
    /// gradient per screened-in sphere, a Jacobian contraction, and a QP solve.
    ///
    /// `ClearanceBarrier::certifiedRegion()` answers a strictly larger question for
    /// strictly less: it returns the whole convex polytope of joint displacements around
    /// `q` that are certified collision-free, and it reads only barrier *values* and
    /// boundaries -- no gradient, no Jacobian, no row, no QP. `certifiedDuration()` and
    /// `safeDuration()` are that polytope intersected with a single ray, which is all the
    /// rollout ever needed because it had already committed to a direction.
    ///
    /// So: keep the polytope, drop the control. An edge is the straight segment, walked
    /// in certified regions -- evaluate, take the largest fraction of what is left that
    /// the region contains, repeat. `safeDuration()` fed the remaining *displacement*
    /// rather than a velocity is exactly that fraction, so the primitive already exists
    /// and is already tested.
    ///
    /// Asking for the region *directly* is what makes the hop cheap. No row is wanted, so
    /// none is built, no gradient is interpolated, no Jacobian is contracted and no
    /// screening sweep runs; the barrier collapses to a batched clearance query. And the
    /// ray restriction is charged once per edge rather than once per hop: the remaining
    /// displacement never turns, so `ClearanceBarrier::travelBound()` is invariant along
    /// the segment and only `ClearanceBarrier::safeScale()` repeats. Measured per hop on
    /// the UR5, against the filtered rollout's 2.5 us:
    ///
    ///     stage                              was       is
    ///     region                             1.42 us   1.09 us   direct, not off an
    ///                                                            `Evaluation`
    ///     ray restriction                    0.39 us   0.09 us   travel hoisted to the
    ///                                                            edge, no per-row divide
    ///     total                              1.81 us   1.18 us
    ///
    /// ### What limits a hop, which is not what you would guess
    ///
    /// With the world eight metres away the region's world slack is 12.7 m and its
    /// *self*-collision slack is 19.8 mm, for a certified radius of 0.12 rad. Free space
    /// therefore costs several evaluations rather than one, and the binding constraint is
    /// the arm against itself. `Robot::selfPairLeverArms()` are maxima over the whole
    /// configuration space, so that 19.8 mm is loose by however far the arm is from its
    /// worst pose: tightening them lengthens every hop this class takes, in open space
    /// most of all. That is the first place to look for more.
    ///
    /// ### What it gives up
    ///
    /// Deflection. Where the rollout slides along an obstacle boundary and carries on,
    /// this stops at the boundary and hands back the prefix. It cannot leave the line.
    ///
    /// ### What that trade measured
    ///
    /// 2000 random 1.5 rad edges per scene, `results/picard_sliding_benchmark`, against
    /// the sequential rollout at kappa 8, step 0.05, `minAdvance` 0.01:
    ///
    ///     scene      time/edge      evaluations/edge   target gap closed
    ///     shelf      0.40x          0.62x              68.9% vs 87.5%
    ///     corridor   0.46x          0.79x              59.4% vs 64.2%
    ///     clutter    0.27x          0.48x              44.2% vs 78.1%
    ///
    /// Per unit of wall time that is close to twice the progress on every scene, and the
    /// gap narrows further on edges aimed straight down the worst barrier's gradient --
    /// the deadlock population, where the rollout's sliding is supposed to pay most and
    /// where it instead buys 75.2% against 71.3% for twice the time.
    ///
    /// Where the rollout stays ahead is dense clutter, which is exactly where sliding is
    /// worth the QP. Neither is dominant, so this is installed rather than substituted:
    /// `FilteredStateSpace::setRolloutPlanner()` takes it and everything downstream --
    /// the ledger, `FilteredMotionValidator`, `ExecutedPath`, RRTConnect's partial
    /// steering -- consumes the same `Rollout` and is unaffected.
    ///
    /// ### Safety
    ///
    /// Every waypoint and every point between two consecutive ones is inside a region
    /// certified from the earlier one, so the whole polyline satisfies `h >= 0` -- a
    /// stronger statement than the rollout's, which holds `h >= 0` at the points the
    /// filter was asked about and leans on the certificate in between. What is *not*
    /// claimed is the CBF envelope: nothing here enforces `dh/dt >= -kappa h`, so a
    /// caller reproducing a continuously-filtered execution wants the rollout, not this.
    /// See `FilteredStateSpace::setSafeHops()`, which gives up the same thing for the
    /// same reason.
    class CertifiedRegionRollout
    {
    public:
        using Configuration = robots::UR5::Configuration;
        using Rollout = FilteredStateSpace::Rollout;

        struct Parameters
        {
            /// Smallest joint-space advance worth another evaluation, in radians.
            ///
            /// Against a contact the regions shrink geometrically and the walk would
            /// otherwise spend its whole budget inching. This is the floor that stops it,
            /// and it is the only real knob: the swept frontier is monotone -- looser
            /// gives more progress per second and more edges that never move at all.
            /// The default is about a fifth of what one filter call of the rollout
            /// covers, which on the benchmark scenes is the knee.
            double minAdvance{0.01};
            /// Evaluations one edge may spend, matching `EarlyTermination`'s default
            /// filter-call budget so the two are charged alike. Zero is unbounded.
            unsigned int maxEvaluations{40};
        };

        struct Statistics
        {
            std::size_t edges{0};
            std::size_t evaluations{0};
            std::size_t completed{0};   ///< walked the whole requested segment
            std::size_t blocked{0};     ///< first region certified nothing
            std::size_t floored{0};     ///< stopped on `minAdvance`
            std::size_t budgeted{0};    ///< stopped on `maxEvaluations`
            std::size_t stillborn{0};   ///< produced no motion at all
            double travel{0.0};
        };

        explicit CertifiedRegionRollout(const ClearanceBarrier &barrier)
          : CertifiedRegionRollout(barrier, Parameters())
        {
        }

        CertifiedRegionRollout(const ClearanceBarrier &barrier, const Parameters &parameters)
          : barrier_(barrier), parameters_(parameters)
        {
            if (parameters.minAdvance < 0.0)
                throw Exception("CertifiedRegionRollout: minAdvance must not be negative");
        }

        /// Walk `from` toward `to` for \p fraction of the horizon. Always returns true:
        /// unlike a speculative proposal there is no fallback to defer to, and a walk
        /// that stops early is a partial edge rather than a failure -- the same thing a
        /// stalled rollout hands back.
        bool plan(const Configuration &from, const Configuration &to, double fraction,
                  Rollout &out) const
        {
            // A region certifies its own closed boundary, where the binding barrier may
            // sit at exactly its bound and the next region is degenerate. Stopping a hair
            // short keeps the chain moving.
            constexpr double shrink = 1.0 - 1e-9;

            const double share = std::clamp(fraction, 0.0, 1.0);
            // The rollout re-aims at `to` every step and so, in free space, covers
            // `share` of the segment. Walking to that same point is what makes the two
            // methods answer the same question.
            const Configuration target = from + share * (to - from);

            out = Rollout();
            out.end = from;
            out.waypoints.push_back(from);
            ++statistics_.edges;

            // Nothing asked for. Answering it with an evaluation would charge the caller
            // for a hop it did not request and append a duplicate waypoint, which is not
            // an edge -- `interpolate()` asks for exactly this when `t` is zero.
            if (share <= 0.0 || (target - from).squaredNorm() <= 0.0)
            {
                ++statistics_.completed;
                ++statistics_.stillborn;
                out.reachedTarget = share >= 1.0;
                return true;
            }

            // Every remaining displacement on this segment is a positive multiple of the
            // direction -- `target - q` contracts by `1 - step` per hop and never turns --
            // so its travel bound is that multiple of the direction's, and the 343 x 6
            // contraction belongs to the edge rather than to the hop. It was half the
            // cost of the ray query. `1 - covered` is the multiple, tracked already.
            const ClearanceBarrier::Values travel = ClearanceBarrier::travelBound(target - from);

            Configuration q = from;
            double covered = 0.0;
            while (covered < 1.0)
            {
                if (parameters_.maxEvaluations > 0 && out.steps >= parameters_.maxEvaluations)
                {
                    out.callBudgetReached = true;
                    ++statistics_.budgeted;
                    break;
                }

                // The region reads values, boundaries and in-bounds-ness and nothing
                // else, so this path costs no gradient interpolation, no Jacobian
                // contraction, no constraint row and no screening sweep -- and it batches
                // the field queries. Measured 1.09 us against 1.33 us for the screened
                // evaluation the region was previously read off, which built rows for
                // every barrier already at or below zero and zero-filled a 343-entry
                // threshold to ask for them.
                const ClearanceBarrier::CertifiedRegion region = barrier_.certifiedRegion(q);
                const Configuration remaining = target - q;
                // `safeScale` answers in shares of the *direction*; `remaining` is
                // `left` of it, so the share of `remaining` is the quotient.
                const double left = 1.0 - covered;
                const double scale = ClearanceBarrier::safeScale(region, travel) / left;

                // Out of the baked field, or a barrier already at its bound. The terminal
                // a blocked filter call is, and charged the same way: work spent, no step.
                if (!(scale > 0.0))
                {
                    out.blocked = 1;
                    ++statistics_.blocked;
                    ++statistics_.evaluations;
                    break;
                }
                ++out.steps;
                ++statistics_.evaluations;

                const double step = std::min(scale * shrink, 1.0);
                const Configuration landing = q + step * remaining;
                const double advance = (landing - q).norm();
                if (step < 1.0 && advance < parameters_.minAdvance)
                {
                    out.stalled = true;
                    ++statistics_.floored;
                    break;
                }

                out.travel += advance;
                out.coarse += advance > 0.0 ? 1u : 0u;
                q = landing;
                out.waypoints.push_back(q);
                covered += step * left;
                if (step >= 1.0)
                {
                    ++statistics_.completed;
                    break;
                }
            }

            out.end = q;
            // `fraction` is a share of the *full* horizon, and the walk covered `covered`
            // of the `share` it was sent after.
            out.fraction = share * covered;
            out.reachedTarget = covered >= 1.0 && share >= 1.0;
            statistics_.travel += out.travel;
            statistics_.stillborn += out.waypoints.size() < 2 ? 1u : 0u;
            return true;
        }

        /// Bound to `FilteredStateSpace::setRolloutPlanner()`.
        FilteredStateSpace::RolloutPlanner planner() const
        {
            return [this](const Configuration &from, const Configuration &to, double fraction,
                          Rollout &out) { return plan(from, to, fraction, out); };
        }

        const Parameters &parameters() const
        {
            return parameters_;
        }

        const Statistics &statistics() const
        {
            return statistics_;
        }

        void resetStatistics() const
        {
            statistics_ = Statistics();
        }

    private:
        const ClearanceBarrier &barrier_;
        Parameters parameters_;
        mutable Statistics statistics_;
    };
}  // namespace ompl::cbf
