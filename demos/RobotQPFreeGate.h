#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
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
    /// then made to satisfy every potentially binding CBF row. No collision checker is
    /// called, and no rollout certificate is issued: the point of the comparison is what
    /// the QP buys, so this reports `certified = 0` and takes one step per call.
    ///
    /// ### Repair by scaling, which needs no solver
    ///
    /// Each active row asks `g.u + kappa h >= 0`. Scaling the control by `alpha` in
    /// [0, 1] leaves a row with `g.u >= 0` satisfied for every `alpha` (the left side is
    /// at least `kappa h >= 0`), and a row with `g.u < 0` satisfied exactly while
    /// `alpha <= kappa h / -(g.u)`. So the minimum of that ratio over the binding rows
    /// satisfies *all* of them at once: one pass, closed form, no iteration and no QP.
    ///
    /// This is a brake, not a steer. It shortens the step along the nominal direction but
    /// cannot deflect around an obstacle, which is what distinguishes it from the QP's
    /// projection -- the QP changes direction to keep moving where scaling can only slow
    /// down. That is the intended gap between this row and the fixed-step QP.
    ///
    /// `OMPL_QPFREE_REPAIR=0` restores the original accept-or-reject behaviour, which is
    /// the published LQR-CBF-RRT* decision and what every result before this change was
    /// measured with. On the UR5 it cost 8.3x the barrier evaluations of `qpFixed` and 7x
    /// the vertices, because a single failing row ended the rollout and RRT-Connect had
    /// to rebuild the lost progress as tree nodes.
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
            repair_ = repairEnabled();
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

            // The largest admissible multiple of the clipped control. Starts at 1 -- the
            // control as asked -- and is pulled down by each row the control moves
            // against. See the class comment for why one pass suffices.
            double scale = 1.0;
            for (Eigen::Index row = 0; row < evaluation_.active; ++row)
            {
                const int constraint = evaluation_.constraint[row];
                const double h = evaluation_.values[constraint];
                if (h < 0.0)
                {
                    // Already inside the margin: no control makes this row hold, so
                    // there is nothing to repair.
                    applied.setZero();
                    return reject();
                }
                const double slope = evaluation_.rows.row(row).dot(applied);
                if (slope < 0.0)
                    scale = std::min(scale, kappa_ * h / -slope);
            }

            if (!repair_)
            {
                // Accept-or-reject: the original decision, kept for the A/B. `scale < 1`
                // is exactly the condition "some row fails at the full control".
                if (scale < 1.0)
                {
                    applied.setZero();
                    return reject();
                }
            }
            else if (scale < 1.0)
            {
                // Clearance is spent along this direction; braking harder cannot buy
                // progress, so stop rather than hand back a control that does nothing.
                if (!(scale > minimumScale))
                {
                    applied.setZero();
                    return reject();
                }
                applied *= scale;
                ++repaired_;
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

        /// How many calls were repaired by scaling rather than passed or rejected.
        std::size_t repaired() const
        {
            return repaired_;
        }

    private:
        /// Below this the scaled control buys no useful travel, so the rollout is better
        /// off stopping than spending a call on it.
        static constexpr double minimumScale = 1e-6;

        Status reject() const
        {
            ++rejected_;
            return Status::Blocked;
        }

        /// Read once. Repair is the default; zero restores the published accept-or-reject
        /// decision, which is what results before this change used.
        static bool repairEnabled()
        {
            static const bool value = []
            {
                const char *v = std::getenv("OMPL_QPFREE_REPAIR");
                return v == nullptr || std::atoi(v) != 0;
            }();
            return value;
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
        mutable std::size_t repaired_{0};
        mutable std::size_t rows_{0};
        bool repair_{true};
    };
}  // namespace ompl::demo
