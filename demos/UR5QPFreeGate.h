#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

#include <Eigen/Core>

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ControlFilter.h>

namespace ompl::demo
{
    /// UR5 adaptation of LQR-CBF-RRT*'s no-QP local safety decision.
    ///
    /// The nominal velocity is clipped to the same actuator/joint box as the QP filter,
    /// then made to satisfy every potentially binding CBF row. No collision checker is
    /// called here, and no rollout certificate is issued, so a hop is always one step.
    ///
    /// ### Repair by scaling, which needs no solver
    ///
    /// Each active row asks `g.u + kappa h >= 0`. Scaling the control by `alpha` in
    /// [0, 1] leaves a row with `g.u >= 0` satisfied for every `alpha` (the left side is
    /// at least `kappa h >= 0`), and a row with `g.u < 0` satisfied exactly while
    /// `alpha <= kappa h / -(g.u)`. So the minimum of that ratio over the binding rows
    /// satisfies *all* of them at once: one pass, closed form, no iteration and no QP.
    ///
    /// This is a brake, not a steer. It shortens the step along the nominal direction
    /// but cannot deflect around an obstacle, which is what distinguishes it from the
    /// QP's projection -- the QP changes direction to keep moving where scaling can only
    /// slow down. That is the intended gap between this row and `qpFixed`.
    ///
    /// Rejection remains for the two cases scaling cannot fix: a row already violated
    /// (`h < 0`, so the state is unsafe before any control is chosen) and `alpha`
    /// collapsing to zero (clearance is already spent, so no multiple of this direction
    /// is admissible).
    ///
    /// `OMPL_QPFREE_REPAIR=0` restores the original accept-or-reject behaviour, which is
    /// what the published LQR-CBF-RRT* decision does and what every result before this
    /// change was measured with. It cost 8.3x the barrier evaluations of `qpFixed`
    /// (10,088 against 1,214) and 7x the vertices, because a single failing row out of
    /// 343 ended the rollout and RRT-Connect had to rebuild the progress as tree nodes.
    class UR5QPFreeGate final : public cbf::ControlFilter
    {
    public:
        using Barrier = cbf::ClearanceBarrier;
        using QPFilter = cbf::CBFControlFilter;

        UR5QPFreeGate(const Barrier &barrier, const QPFilter::Parameters &parameters)
          : barrier_(barrier), parameters_(parameters),
            decreaseRates_(barrier.decreaseRates(parameters.maxSpeed)),
            repair_(repairEnabled())
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

            // The largest admissible multiple of the clipped control. Starts at 1 -- the
            // control as asked -- and is pulled down by each row that the control moves
            // against. See the class comment for why one pass suffices.
            double scale = 1.0;
            for (Eigen::Index row = 0; row < evaluation_.active; ++row)
            {
                const int constraint = evaluation_.constraint[row];
                const double h = evaluation_.values[constraint];
                if (h < 0.0)
                {
                    // Already inside the margin: no choice of control makes this row
                    // hold, so there is nothing to repair.
                    applied.setZero();
                    return reject();
                }
                const double slope = evaluation_.rows.row(row).dot(applied);
                if (slope < 0.0)
                    scale = std::min(scale, parameters_.kappa * h / -slope);
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
        QPFilter::Parameters parameters_;
        Barrier::Values decreaseRates_;
        mutable Barrier::Values threshold_;
        mutable Barrier::Evaluation evaluation_;
        mutable double cachedHorizon_{-1.0};
        mutable std::size_t calls_{0};
        mutable std::size_t rejected_{0};
        mutable std::size_t repaired_{0};
        mutable std::size_t rows_{0};
        bool repair_;
    };
}  // namespace ompl::demo
