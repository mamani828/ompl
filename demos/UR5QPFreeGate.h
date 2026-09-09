#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ControlFilter.h>

namespace ompl::demo
{
    /// UR5 adaptation of LQR-CBF-RRT*'s no-QP local safety decision.
    ///
    /// The nominal velocity is clipped to the same actuator/joint box as the QP
    /// filter, then accepted only when every potentially binding CBF row passes.
    /// A failed row stops the rollout; this class never repairs a control and never
    /// issues a rollout certificate. No collision checker is called here.
    class UR5QPFreeGate final : public cbf::ControlFilter
    {
    public:
        using Barrier = cbf::ClearanceBarrier;
        using QPFilter = cbf::CBFControlFilter;

        UR5QPFreeGate(const Barrier &barrier, const QPFilter::Parameters &parameters)
          : barrier_(barrier), parameters_(parameters),
            decreaseRates_(barrier.decreaseRates(parameters.maxSpeed))
        {
            threshold_.setZero();
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &applied) const override
        {
            ++calls_;
            applied.setZero();
            if (!(duration > 0.0))
                return reject();

            Control lower = -parameters_.maxSpeed.cwiseAbs();
            Control upper = parameters_.maxSpeed.cwiseAbs();
            if (parameters_.respectJointLimits)
            {
                lower = lower.cwiseMax((robots::UR5::lowerBounds() - q) / duration);
                upper = upper.cwiseMin((robots::UR5::upperBounds() - q) / duration);
            }
            applied = nominal.cwiseMax(lower).cwiseMin(upper);

            // Upstream selects nearby obstacles. This is the sound UR5 counterpart:
            // a skipped row is satisfied by every control in the shared control box,
            // so screening cannot change the gate's accept/reject decision.
            const double horizon = parameters_.kappa > 0.0
                                       ? std::max(duration, 1.0 / parameters_.kappa)
                                       : std::numeric_limits<double>::infinity();
            if (horizon != cachedHorizon_)
            {
                threshold_ = decreaseRates_ * horizon;
                cachedHorizon_ = horizon;
            }
            barrier_.evaluateScreened(q, threshold_, evaluation_);
            rows_ += static_cast<std::size_t>(evaluation_.active);
            if (!evaluation_.inBounds)
            {
                applied.setZero();
                return reject();
            }

            for (Eigen::Index row = 0; row < evaluation_.active; ++row)
            {
                const int constraint = evaluation_.constraint[row];
                const double h = evaluation_.values[constraint];
                if (h < 0.0 ||
                    evaluation_.rows.row(row).dot(applied) + parameters_.kappa * h < 0.0)
                {
                    applied.setZero();
                    return reject();
                }
            }

            return applied.isApprox(nominal, 0.0) ? Status::Unchanged : Status::Filtered;
        }

        const char *name() const override
        {
            return "ur5-qp-free-cbf-gate";
        }

        std::size_t calls() const
        {
            return calls_;
        }

        std::size_t rejected() const
        {
            return rejected_;
        }

        double meanRows() const
        {
            return calls_ ? static_cast<double>(rows_) / static_cast<double>(calls_) : 0.0;
        }

    private:
        Status reject() const
        {
            ++rejected_;
            return Status::Blocked;
        }

        const Barrier &barrier_;
        QPFilter::Parameters parameters_;
        Barrier::Values decreaseRates_;
        mutable Barrier::Values threshold_;
        mutable Barrier::Evaluation evaluation_;
        mutable double cachedHorizon_{-1.0};
        mutable std::size_t calls_{0};
        mutable std::size_t rejected_{0};
        mutable std::size_t rows_{0};
    };
}  // namespace ompl::demo
