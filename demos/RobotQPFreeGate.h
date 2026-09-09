#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <ompl/cbf/ControlFilter.h>
#include <ompl/cbf/RobotCBFControlFilter.h>
#include <ompl/cbf/RobotClearanceBarrier.h>

namespace ompl::demo
{
    /// LQR-CBF-RRT*'s no-QP local safety decision, for any robot the templated
    /// `cbf::RobotClearanceBarrier` serves. The generic counterpart of `UR5QPFreeGate`,
    /// which does the same thing against the hand-written UR5 barrier.
    ///
    /// The nominal velocity is clipped to the same actuator/joint box the QP filter uses,
    /// then *accepted or rejected* -- never repaired. A single failing CBF row stops the
    /// rollout. No collision checker is called, and no rollout certificate is issued: the
    /// point of the comparison is what the QP buys, so this reports `certified = 0` and
    /// takes one step per call.
    ///
    /// ### Why screening cannot change the decision
    ///
    /// Rows are screened at `decreaseRates * max(dt, 1/kappa)`, the same threshold the QP
    /// filter uses. A skipped row has clearance above what any control in the shared
    /// control box could consume over that horizon, so it is satisfied by every control
    /// the gate could possibly accept -- dropping it cannot flip an accept to a reject or
    /// the other way. This is the soundness argument `UR5QPFreeGate` documents; it is
    /// unchanged here.
    template <typename Robot>
    class RobotQPFreeGate final : public cbf::RobotControlFilter<Robot>
    {
    public:
        using Base = cbf::RobotControlFilter<Robot>;
        using Configuration = typename Base::Configuration;
        using Control = typename Base::Control;
        using Status = typename Base::Status;
        using Barrier = cbf::RobotClearanceBarrier<Robot>;
        using QPFilter = cbf::RobotCBFControlFilter<Robot>;

        static constexpr int nJoints = static_cast<int>(Robot::nJoints);
        static constexpr int nBaseJoints = Barrier::nBaseJoints;

        /// \p kappa is the decay rate the rows are tested at; it defaults to the QP
        /// filter's own so the two rows of a comparison are charged alike.
        explicit RobotQPFreeGate(const Barrier &barrier, double kappa = QPFilter::kappa)
          : barrier_(barrier), kappa_(kappa), lowerPosition_(Robot::lowerBounds()),
            upperPosition_(Robot::upperBounds())
        {
            barrier_.decreaseRates(QPFilter::maxSpeed(), decreaseRates_);
            threshold_.setZero();
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &applied) const override
        {
            ++calls_;
            applied.setZero();
            if (!(duration > 0.0))
                return reject();

            const Control maxSpeed = QPFilter::maxSpeed();
            Control lower = -maxSpeed;
            Control upper = maxSpeed;
            for (int j = 0; j < nJoints; ++j)
            {
                if constexpr (nBaseJoints >= 3)
                    if (j == 2)
                        continue;  // periodic base yaw: a fixed position limit is meaningless
                lower[j] = std::max(lower[j], (lowerPosition_[j] - q[j]) / duration);
                upper[j] = std::min(upper[j], (upperPosition_[j] - q[j]) / duration);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }
            applied = nominal.cwiseMax(lower).cwiseMin(upper);

            const double horizon =
                kappa_ > 0.0 ? std::max(duration, 1.0 / kappa_) : std::numeric_limits<double>::infinity();
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
                if (h < 0.0 || evaluation_.rows.row(row).dot(applied) + kappa_ * h < 0.0)
                {
                    applied.setZero();
                    return reject();
                }
            }
            return applied.isApprox(nominal, 0.0) ? Status::Unchanged : Status::Filtered;
        }

        const char *name() const override
        {
            return "robot-qp-free-cbf-gate";
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
        double kappa_;
        Configuration lowerPosition_, upperPosition_;
        typename Barrier::Values decreaseRates_;
        mutable typename Barrier::Values threshold_;
        mutable typename Barrier::Evaluation evaluation_;
        mutable double cachedHorizon_{-1.0};
        mutable std::size_t calls_{0};
        mutable std::size_t rejected_{0};
        mutable std::size_t rows_{0};
    };
}  // namespace ompl::demo
