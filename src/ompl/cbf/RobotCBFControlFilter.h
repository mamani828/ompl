#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>

#include <Eigen/Core>
#include <qpmad/solver.h>

#include <ompl/cbf/ControlFilter.h>
#include <ompl/cbf/RobotClearanceBarrier.h>

namespace ompl::cbf
{
    /// A control barrier function safety filter for a Reachy2-family robot,
    /// solved as a small QP against a `RobotClearanceBarrier<Robot>`. See that
    /// class for the barrier this enforces, and `ompl::cbf::CBFControlFilter`
    /// for the UR5-family sibling this design deliberately does not merge with
    /// (different robot-model interface -- see `RobotClearanceBarrier`'s class
    /// comment).
    ///
    /// The nominal control is projected onto the set of controls that keep
    /// every screened-in barrier's clearance from decaying too fast:
    ///
    ///     minimize    0.5 (u - uNom)^T W (u - uNom)
    ///     subject to  (dh_i/dq) u  >=  -kappa * (h_i(q) - buffer)
    ///                 lower <= u <= upper
    ///
    /// `kappa` is a decay rate in 1/s, so the row is the continuous-time condition
    /// `dh/dt >= -kappa h` and carries no step length; see
    /// `ompl::cbf::CBFControlFilter` for why that replaced the per-step form.
    ///
    /// ### Why the objective is speed-weighted, not identity
    ///
    /// The fixed-base original used a plain identity cost, `W = I`. That is not
    /// a sound default once a robot mixes units in one control vector -- a
    /// planar base's `[x, y, yaw]` prefix is metres and radians per second,
    /// alongside the arm's radians per second -- because an unweighted
    /// least-squares projection would treat a millimetre of unwanted base drift
    /// as equal in cost to a milliradian of unwanted arm motion. This class
    /// instead weights each joint by `1 / maxSpeed_j^2`, so the cost is
    /// expressed in "fraction of that joint's own speed limit" rather than raw
    /// units.
    ///
    /// This is not a behaviour change for a robot whose `maxSpeed()` happens to
    /// be uniform across every joint -- which the fixed-base `Reachy2` case is,
    /// since its own `velocityLimits()` is a flat placeholder that `maxSpeed()`
    /// clamps to the same 1.2 rad/s everywhere. A uniform diagonal weight is a
    /// positive scalar multiple of the identity, and scaling a QP's objective by
    /// a positive constant never changes its argmin (the feasible region and
    /// constraints are unaffected) -- so this formula reproduces the old
    /// fixed-base filter's output exactly, and is what a mixed-unit mobile base
    /// actually needs. See `test_robot_cbf_control_filter.cpp`'s numerical
    /// equivalence tests for the empirical check, rather than resting on the
    /// argument alone.
    ///
    /// ### The integration buffer
    ///
    /// \p integrationBuffer is subtracted (floored at zero) from every screened
    /// barrier value before it is used -- both when deciding which rows to
    /// screen in, and inside the QP's row lower bound -- reserving that much
    /// clearance the filter will never spend. It defaults to zero, which is a
    /// no-op reproducing the fixed-base original's behaviour exactly; the
    /// mobile-base original reserved 1 mm here, having found empirically that a
    /// wider configuration space wants a little more room before the
    /// discrete-step linearisation is trusted.
    ///
    /// ### Why this file is header-only
    ///
    /// `ompl::cbf::CBFControlFilter` hides qpmad from its header behind a pimpl
    /// (`struct Solver;` forward-declared, defined only in `CBFControlFilter.cpp`,
    /// held through a `std::unique_ptr`) -- which works because there is exactly
    /// one concrete size to hide. A class *template* has no single compiled
    /// `.cpp`: `qpmad::SolverTemplate<double, nJoints, ...>` must be a plain
    /// value member (fixed-size, stack-allocated) sized per `Robot`, so its
    /// definition has to be visible wherever this template is instantiated.
    /// Pimpl-ing it would need an explicit-instantiation `.cpp` per `Robot`,
    /// defeating the point of a template -- so this header includes
    /// `<qpmad/solver.h>` directly, which is otherwise kept out of installed
    /// `ompl` headers (see `src/ompl/CMakeLists.txt`). Harmless as long as
    /// nothing includes this header when `OMPL_HAVE_QPMAD` is off, which is how
    /// every current caller is already gated.
    template <typename Robot>
    class RobotCBFControlFilter : public RobotControlFilter<Robot>
    {
    public:
        using Base = RobotControlFilter<Robot>;
        using typename Base::Configuration;
        using typename Base::Control;
        using Status = typename Base::Status;
        using Barrier = RobotClearanceBarrier<Robot>;

        static constexpr int nJoints = Barrier::nJoints;
        static constexpr int nBaseJoints = Barrier::nBaseJoints;
        /// What the old per-step decay of 0.6 amounted to at this family's 0.02 s
        /// integration step, now stated as a rate: `dh/dt >= -kappa h`.
        static constexpr double kappa = 30.0;

        /// Position bounds default to the robot's compiled-in joint limits, and
        /// `integrationBuffer` defaults to zero. This is the fixed-base
        /// original's single-argument constructor.
        explicit RobotCBFControlFilter(const Barrier &barrier)
          : RobotCBFControlFilter(barrier, Robot::lowerBounds(), Robot::upperBounds())
        {
        }

        /// As above, with an explicit position box -- e.g. a CLI-narrowed mobile
        /// base workspace -- overriding the compiled-in bounds; buffer still
        /// defaults to zero.
        RobotCBFControlFilter(const Barrier &barrier, const Configuration &lower, const Configuration &upper)
          : RobotCBFControlFilter(barrier, lower, upper, 0.0)
        {
        }

        /// Full form. See the class comment for what \p integrationBuffer does.
        RobotCBFControlFilter(const Barrier &barrier, const Configuration &lower, const Configuration &upper,
                              double integrationBuffer)
          : barrier_(barrier), lowerPosition_(lower), upperPosition_(upper), integrationBuffer_(integrationBuffer)
        {
            rowUpper_.setConstant(std::numeric_limits<double>::infinity());
            barrier_.decreaseRates(maxSpeed(), decreaseRates_);
            const Configuration speed = maxSpeed();
            inverseSquaredSpeed_ = speed.cwiseInverse().cwiseProduct(speed.cwiseInverse());
        }

        /// Legacy safety clamp for robots (`Reachy2`) whose own `velocityLimits()`
        /// is an unrealistic placeholder; a no-op for robots (the mobile
        /// wrapper) that already bake in real per-joint limits at or below
        /// 1.2 rad/s or m/s.
        static Configuration maxSpeed()
        {
            return Robot::velocityLimits().cwiseMin(Configuration::Constant(1.2));
        }

        Status filter(const Configuration &q, const Configuration &nominal, double dt,
                      Configuration &filtered) const override
        {
            double ignored = 0.0;
            return filter(q, nominal, dt, filtered, ignored);
        }

        Status filter(const Configuration &q, const Configuration &nominal, double dt, Configuration &filtered,
                      double &certified) const override
        {
            ++calls_;
            certified = 0.0;
            if (dt <= 0.0)
            {
                filtered.setZero();
                return Status::Blocked;
            }

            // rate * max(dt, 1/kappa): the first horizon keeps a skipped row non-negative
            // across the step, the second makes it provably non-binding in the QP. See
            // ClearanceBarrier::evaluateScreened().
            threshold_ = decreaseRates_ * std::max(dt, 1.0 / kappa);
            threshold_.array() += integrationBuffer_;  // a no-op when integrationBuffer_ == 0
            barrier_.evaluateScreened(q, threshold_, evaluation_);
            activeRows_ += static_cast<std::size_t>(evaluation_.active);
            if (!evaluation_.inBounds)
            {
                filtered.setZero();
                return Status::Blocked;
            }

            Configuration lower = -maxSpeed();
            Configuration upper = maxSpeed();
            for (int j = 0; j < nJoints; ++j)
            {
                if constexpr (nBaseJoints >= 3)
                    if (j == 2)
                        continue;  // periodic base yaw: a fixed position limit is meaningless
                lower[j] = std::max(lower[j], (lowerPosition_[j] - q[j]) / dt);
                upper[j] = std::min(upper[j], (upperPosition_[j] - q[j]) / dt);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }

            const Eigen::Index active = evaluation_.active;
            if (active == 0)
                filtered = nominal.cwiseMax(lower).cwiseMin(upper);
            else
            {
                ++qpCalls_;
                hessian_.setZero();
                hessian_.diagonal() = inverseSquaredSpeed_;
                objective_ = -inverseSquaredSpeed_.cwiseProduct(nominal);
                for (Eigen::Index row = 0; row < active; ++row)
                    rowLower_[row] =
                        -kappa * (evaluation_.values[evaluation_.constraint[row]] - integrationBuffer_);
                try
                {
                    const auto status = solver_.solve(filtered, hessian_, objective_, lower, upper,
                                                       evaluation_.rows.topRows(active), rowLower_.head(active),
                                                       rowUpper_.head(active));
                    if (status != Solver::OK)
                    {
                        filtered.setZero();
                        return Status::Blocked;
                    }
                }
                catch (const std::exception &)
                {
                    filtered.setZero();
                    return Status::Blocked;
                }
            }

            certified = barrier_.certifiedDuration(evaluation_, filtered, kappa, integrationBuffer_);
            for (int j = 0; j < nJoints; ++j)
            {
                if constexpr (nBaseJoints >= 3)
                    if (j == 2)
                        continue;
                if (filtered[j] == 0.0)
                    continue;
                const double room = (filtered[j] > 0.0 ? upperPosition_[j] : lowerPosition_[j]) - q[j];
                certified = std::min(certified, std::max(room / filtered[j], 0.0));
            }
            return (filtered - nominal).norm() <= 1e-12 ? Status::Unchanged : Status::Filtered;
        }

        const char *name() const override
        {
            return "robot-cbf-qp";
        }

        std::size_t calls() const
        {
            return calls_;
        }

        std::size_t qpCalls() const
        {
            return qpCalls_;
        }

        double meanActiveRows() const
        {
            return calls_ > 0 ? static_cast<double>(activeRows_) / calls_ : 0.0;
        }

    private:
        using Solver = qpmad::SolverTemplate<double, nJoints, 1, Barrier::maxConstraints>;

        const Barrier &barrier_;
        Configuration lowerPosition_, upperPosition_;
        double integrationBuffer_;
        mutable Solver solver_;
        mutable Eigen::Matrix<double, nJoints, nJoints> hessian_;
        Configuration inverseSquaredSpeed_;
        mutable Configuration objective_;
        mutable typename Barrier::Values rowLower_, rowUpper_, threshold_;
        typename Barrier::Values decreaseRates_;
        mutable typename Barrier::Evaluation evaluation_;
        mutable std::size_t calls_{0};
        mutable std::size_t qpCalls_{0};
        mutable std::size_t activeRows_{0};
    };
}  // namespace ompl::cbf
