#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include <ompl/sdf/GridSDF.h>

namespace ompl::cbf
{
    namespace detail
    {
        /// How many of `Robot::Configuration`'s leading entries are a rigid planar
        /// base pose rather than an arm joint. Zero unless `Robot::nBaseJoints`
        /// exists, in which case that value is used -- the same `void_t` dispatch
        /// `RobotConfigurationOperationsImpl` (ConfigurationOperations.h) uses to
        /// pick up a robot's optional `ConfigurationOperations`, just pulling a
        /// value instead of a type.
        template <typename Robot, typename = void>
        struct RobotBaseJoints
        {
            static constexpr std::size_t value = 0;
        };

        template <typename Robot>
        struct RobotBaseJoints<Robot, std::void_t<decltype(Robot::nBaseJoints)>>
        {
            static constexpr std::size_t value = Robot::nBaseJoints;
        };
    }  // namespace detail

    /// The barrier function a CBF steering step constrains, for a spherized
    /// Reachy2-family robot -- a fixed-base arm, or the same arm behind a planar
    /// holonomic base -- in a baked workspace signed distance field.
    ///
    /// For each collision sphere i with radius r_i and center p_i(q):
    ///
    ///     h_i(q) = d(p_i(q)) - r_i - worldMargin
    ///
    /// and, by the chain rule through the sphere-center Jacobian,
    ///
    ///     dh_i/dq = (dp_i/dq)^T grad d(p_i)
    ///
    /// `evaluateScreened()` returns both, for whichever spheres the caller's
    /// threshold keeps: the values say how much clearance each sphere has, and
    /// the rows are the linear constraint rows a QP steering step needs. The
    /// robot is safe exactly when every h_i >= 0.
    ///
    /// The arm-against-itself family works the same way, one row per pair in
    /// `Robot::selfPairs()`:
    ///
    ///     h_ab(q) = |p_a(q) - p_b(q)| - r_a - r_b - selfMargin
    ///
    /// Both families share one flat index -- `i < nSpheres` is sphere i against
    /// the world, `nSpheres + p` is pair p -- so everything downstream is
    /// indifferent to which kind a row is. `worldMargin`/`selfMargin` are the
    /// same constants `ReachyBarrier`/`MobileBarrier` used; see
    /// `ompl::cbf::ClearanceBarrier`'s class comment for the fuller rationale
    /// (sphere under-coverage, SDF discretization, step linearization) a margin
    /// has to absorb -- the argument is the same here, just against Reachy2's
    /// numbers rather than the UR5's.
    ///
    /// This class is deliberately not merged with `ompl::cbf::ClearanceBarrier`,
    /// which serves `robots::UR5` through a different, heavier interface
    /// (precomputed static lever-arm/self-pair tables). What this class asks of
    /// a robot instead -- `steps()`, `spheres()`, `selfPairs()`, `sphereCenter()`,
    /// `sphereJacobian()` -- is the lighter, chain-walking interface `Reachy2`
    /// already provides; unifying the two would mean retrofitting one robot
    /// model onto the other's interface, which is a separate, larger effort.
    ///
    /// ### Base joints and the yaw assumption
    ///
    /// A robot may prepend a rigid planar base pose (`[x, y, yaw]`) to its arm
    /// joints -- `HolonomicMobileManipulator<ArmRobot>` does, via
    /// `nBaseJoints == 3`. This class special-cases exactly that: base
    /// translation moves every sphere at unit lever-arm, base rotation moves a
    /// sphere no faster than its full reach from `base_link`, and rigid base
    /// motion cancels exactly in any self-pair separation. It assumes at most
    /// one such prefix, with yaw always at configuration index 2 -- true for
    /// every robot in this codebase today -- and enforces it with a
    /// `static_assert` rather than merely documenting it, so a future base
    /// shape fails loudly at compile time instead of silently misbehaving.
    ///
    /// ### Why `pairGradient` always pays for two extra Jacobians
    ///
    /// A self-pair's row is `(dp_a/dq - dp_b/dq)^T` dotted with the pair's unit
    /// separation vector. The fixed-base original computed this by hand, walking
    /// only the joints between the two spheres' frames using `Kinematics::axis`/
    /// `origin` arrays. Those arrays do not exist on
    /// `HolonomicMobileManipulator::Kinematics` (which only wraps the arm's own
    /// kinematics plus a base origin), so that fast path cannot generalize. This
    /// class instead builds the row from two full `sphereJacobian()` calls,
    /// which every robot already provides. That is strictly more work per active
    /// self-pair row than the old fixed-base fast path -- accepted here as a
    /// correctness-over-micro-performance tradeoff, not optimized further.
    template <typename Robot>
    class RobotClearanceBarrier
    {
    public:
        using Configuration = typename Robot::Configuration;
        using Kinematics = typename Robot::Kinematics;

        static constexpr int nJoints = static_cast<int>(Robot::nJoints);
        static constexpr int nSpheres = static_cast<int>(Robot::nSpheres);
        static constexpr int nSelfPairs = static_cast<int>(Robot::nSelfPairs);
        static constexpr int nBaseJoints = static_cast<int>(detail::RobotBaseJoints<Robot>::value);
        static_assert(nBaseJoints == 0 || nBaseJoints == 3,
                      "RobotClearanceBarrier assumes at most one planar [x, y, yaw] base "
                      "prefix, with yaw at configuration index 2 -- see the class comment.");

        static constexpr double worldMargin = 0.010;
        static constexpr double selfMargin = 0.005;
        static constexpr int maxConstraints = nSpheres + nSelfPairs;

        using Values = Eigen::Matrix<double, maxConstraints, 1>;
        /// Row i is dh_i/dq -- the constraint row barrier i contributes.
        using Rows = Eigen::Matrix<double, maxConstraints, nJoints>;
        /// Defined locally rather than reused from `Robot::SphereCenters`: not
        /// every robot (`Reachy2`) defines that typedef, and this is exactly
        /// what it would say.
        using Centers = Eigen::Matrix<double, 3, nSpheres>;

        struct Evaluation
        {
            Values values;   ///< h_i(q), for every screened-in constraint
            Rows rows;       ///< dh_i/dq -- only the first `active` are filled
            /// Which barrier each of the first `active` rows belongs to.
            Eigen::Matrix<int, maxConstraints, 1> constraint;
            /// How far sphere i's centre may travel before leaving the SDF's
            /// box. World spheres only; see `certifiedDuration()`.
            Eigen::Matrix<double, nSpheres, 1> boundary;
            int active{0};        ///< how many rows of `rows` are meaningful
            bool inBounds{true};  ///< were all sphere centers inside the SDF's box?
        };

        /// Neither \p robot nor \p field is copied; both must outlive this
        /// object. \p reference is the configuration used to decide which
        /// self-pairs need a row at all: a pair whose two spheres share the
        /// same influence mask moves as one rigid body, so its separation is
        /// invariant and needs no CBF row regardless of configuration.
        RobotClearanceBarrier(const Robot &robot, const sdf::GridSDF &field, const Configuration &reference)
          : robot_(robot), field_(field)
        {
            buildLeverBounds();
            const auto kin = robot_.kinematics(reference);
            for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
            {
                const auto pair = Robot::selfPairs()[p];
                const auto &a = Robot::spheres()[pair.a];
                const auto &b = Robot::spheres()[pair.b];
                const double gap = (Robot::sphereCenter(kin, pair.a) - Robot::sphereCenter(kin, pair.b)).norm() -
                                    a.radius - b.radius;
                // Equal influence masks mean every active joint moves both spheres as
                // one rigid body, so their distance is invariant and needs no CBF row.
                if (a.influence != b.influence && gap > selfMargin + 0.02)
                    selfPairs_.push_back(p);
            }
            buildPairLeverBounds();
        }

        std::size_t constraintCount() const
        {
            return static_cast<std::size_t>(nSpheres) + selfPairs_.size();
        }

        std::size_t enabledSelfPairs() const
        {
            return selfPairs_.size();
        }

        /// The index into `Robot::selfPairs()` behind enabled self-pair
        /// \p enabledIndex (i.e. constraint flat-index `nSpheres + enabledIndex`).
        /// For tests and diagnostics that want to relate a row back to its
        /// physical pair; nothing in the filter path needs this.
        std::size_t selfPairIndex(std::size_t enabledIndex) const
        {
            return selfPairs_[enabledIndex];
        }

        /// How fast each barrier can possibly fall, per unit time, given a
        /// per-joint speed limit. See `ompl::cbf::ClearanceBarrier::decreaseRates()`
        /// for the Lipschitz argument that makes screening against this sound;
        /// it applies unchanged here.
        void decreaseRates(const Configuration &speed, Values &rates) const
        {
            rates.setZero();
            const Configuration absolute = speed.cwiseAbs();
            rates.template head<nSpheres>() = field_.maxGradientNorm() * (leverBounds_ * absolute);
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
                rates[static_cast<Eigen::Index>(nSpheres) + static_cast<Eigen::Index>(p)] =
                    pairLeverBounds_.row(static_cast<Eigen::Index>(p)).dot(absolute);
        }

        /// Barrier values for every constraint, but rows only for the ones
        /// whose clearance is at or below \p threshold. See
        /// `ClearanceBarrier::evaluateScreened()` for what a caller gives up by
        /// screening; the tradeoff is identical here.
        void evaluateScreened(const Configuration &q, const Values &threshold, Evaluation &out) const
        {
            const auto kin = robot_.kinematics(q);
            Centers centers;
            out.active = 0;
            out.inBounds = true;
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers.col(static_cast<Eigen::Index>(i)) = Robot::sphereCenter(kin, i);
                out.inBounds = out.inBounds && field_.inBounds(centers.col(static_cast<Eigen::Index>(i)));
                out.boundary[static_cast<Eigen::Index>(i)] = boundaryClearance(centers.col(static_cast<Eigen::Index>(i)));
                out.values[static_cast<Eigen::Index>(i)] =
                    field_.distance(centers.col(static_cast<Eigen::Index>(i))) - Robot::spheres()[i].radius - worldMargin;
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const auto pair = Robot::selfPairs()[selfPairs_[p]];
                out.values[static_cast<Eigen::Index>(nSpheres) + static_cast<Eigen::Index>(p)] =
                    (centers.col(pair.a) - centers.col(pair.b)).norm() - Robot::spheres()[pair.a].radius -
                    Robot::spheres()[pair.b].radius - selfMargin;
            }
            const std::size_t count = constraintCount();
            for (std::size_t flat = 0; flat < count; ++flat)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(flat);
                if (out.values[index] > threshold[index])
                    continue;
                const Eigen::Index row = out.active++;
                out.constraint[row] = static_cast<int>(flat);
                if (flat < static_cast<std::size_t>(nSpheres))
                    out.rows.row(row) = field_.gradient(centers.col(index)).transpose() * Robot::sphereJacobian(kin, flat);
                else
                    out.rows.row(row) = pairGradient(kin, centers, flat - static_cast<std::size_t>(nSpheres)).transpose();
            }
        }

        /// How long the constant control \p control may be applied from the
        /// configuration \p evaluation was taken at before any row could bind,
        /// i.e. the span over which a filter enforcing this barrier is provably
        /// a no-op. \p buffer is subtracted (floored at zero) from every barrier
        /// value before it is scaled by \p gamma -- see
        /// `RobotCBFControlFilter`'s "integration buffer" doc for why a caller
        /// might want that, and why it defaults to zero (a no-op, reproducing
        /// the original fixed-base behaviour exactly). See
        /// `ClearanceBarrier::certifiedDuration()` for the full Lipschitz
        /// argument this specializes.
        double certifiedDuration(const Evaluation &evaluation, const Configuration &control, double gamma,
                                 double buffer = 0.0) const
        {
            const Configuration speed = control.cwiseAbs();
            const auto worldTravel = (leverBounds_ * speed).eval();
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            double duration = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                if (worldTravel[index] <= 0.0)
                    continue;
                const double allowance = std::min(gamma * std::max(evaluation.values[index] - buffer, 0.0) / lipschitz,
                                                  evaluation.boundary[index]);
                duration = std::min(duration, allowance / worldTravel[index]);
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const double travel = pairLeverBounds_.row(static_cast<Eigen::Index>(p)).dot(speed);
                if (travel > 0.0)
                    duration = std::min(duration,
                        gamma * std::max(evaluation.values[static_cast<Eigen::Index>(nSpheres) +
                                                            static_cast<Eigen::Index>(p)] - buffer, 0.0) / travel);
            }
            return std::max(duration, 0.0);
        }

        /// The full (unscreened) safety check: worst world clearance and worst
        /// enabled self-pair clearance, both via \p world / \p self if given.
        bool safe(const Configuration &q, double *world = nullptr, double *self = nullptr) const
        {
            const auto kin = robot_.kinematics(q);
            Centers centers;
            bool inBounds = true;
            double worstWorld = std::numeric_limits<double>::infinity();
            double worstSelf = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers.col(static_cast<Eigen::Index>(i)) = Robot::sphereCenter(kin, i);
                const Eigen::Vector3d center = centers.col(static_cast<Eigen::Index>(i));
                inBounds = inBounds && field_.inBounds(center);
                worstWorld = std::min(worstWorld, field_.distance(center) - Robot::spheres()[i].radius - worldMargin);
            }
            for (const std::size_t source : selfPairs_)
            {
                const auto pair = Robot::selfPairs()[source];
                worstSelf = std::min(worstSelf, (centers.col(pair.a) - centers.col(pair.b)).norm() -
                                                     Robot::spheres()[pair.a].radius - Robot::spheres()[pair.b].radius -
                                                     selfMargin);
            }
            if (world)
                *world = worstWorld;
            if (self)
                *self = worstSelf;
            return inBounds && worstWorld >= 0.0 && worstSelf >= 0.0;
        }

    private:
        double boundaryClearance(const Eigen::Vector3d &point) const
        {
            return std::min((point - field_.bounds().min()).minCoeff(), (field_.bounds().max() - point).minCoeff());
        }

        /// One algorithm for both robot shapes: with `nBaseJoints == 0` the two
        /// `if constexpr` blocks are dead code and this reduces exactly to the
        /// fixed-base original; with `nBaseJoints == 3` it reduces exactly to
        /// the mobile-base original (base translation at unit lever-arm, arm
        /// columns offset by `nBaseJoints`, yaw lever-arm equal to full reach).
        void buildLeverBounds()
        {
            leverBounds_.setZero();
            const auto &steps = Robot::steps();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index row = static_cast<Eigen::Index>(i);
                if constexpr (nBaseJoints > 0)
                {
                    // Rigid planar translation moves every point at exactly unit rate.
                    leverBounds_(row, 0) = 1.0;
                    leverBounds_(row, 1) = 1.0;
                }

                int link = Robot::spheres()[i].link;
                double reach = Eigen::Vector3d(Robot::spheres()[i].center.data()).norm();
                while (link > 0)
                {
                    const auto &step = steps[static_cast<std::size_t>(link - 1)];
                    if (step.active >= 0)
                        leverBounds_(row, nBaseJoints + step.active) = reach;
                    reach += Eigen::Vector3d(step.xyz.data()).norm();
                    link = step.parent;
                }

                if constexpr (nBaseJoints >= 3)
                    // Rotation about the base z axis (column 2) moves a point no
                    // faster than its full chain distance from base_link.
                    leverBounds_(row, 2) = reach;
            }
        }

        /// Same gating as `buildLeverBounds()`: `armJoints` is `ArmRobot::nJoints`
        /// whenever a base is present, since `HolonomicMobileManipulator`
        /// defines `nJoints = nBaseJoints + ArmRobot::nJoints`.
        void buildPairLeverBounds()
        {
            pairLeverBounds_.setZero();
            constexpr int armJoints = nJoints - nBaseJoints;
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const auto pair = Robot::selfPairs()[selfPairs_[p]];
                const auto maskA = Robot::spheres()[pair.a].influence;
                const auto maskB = Robot::spheres()[pair.b].influence;
                for (int j = 0; j < armJoints; ++j)
                {
                    const bool movesA = (maskA & (1u << j)) != 0;
                    const bool movesB = (maskB & (1u << j)) != 0;
                    if (movesA != movesB)
                        pairLeverBounds_(static_cast<Eigen::Index>(p), nBaseJoints + j) =
                            leverBounds_(static_cast<Eigen::Index>(movesA ? pair.a : pair.b), nBaseJoints + j);
                }
                if constexpr (nBaseJoints > 0)
                    // All planar-base contributions are rigid and cancel exactly.
                    pairLeverBounds_.template block<1, nBaseJoints>(static_cast<Eigen::Index>(p), 0).setZero();
            }
        }

        Configuration pairGradient(const Kinematics &kin, const Centers &centers, std::size_t enabledPair) const
        {
            const auto pair = Robot::selfPairs()[selfPairs_[enabledPair]];
            const Eigen::Vector3d delta = centers.col(pair.a) - centers.col(pair.b);
            Configuration row = Configuration::Zero();
            if (delta.norm() > 1e-12)
                row = (Robot::sphereJacobian(kin, pair.a) - Robot::sphereJacobian(kin, pair.b)).transpose() *
                      delta.normalized();
            if constexpr (nBaseJoints > 0)
                row.template head<nBaseJoints>().setZero();
            return row;
        }

        const Robot &robot_;
        const sdf::GridSDF &field_;
        std::vector<std::size_t> selfPairs_;
        Eigen::Matrix<double, nSpheres, nJoints> leverBounds_;
        Eigen::Matrix<double, nSelfPairs, nJoints> pairLeverBounds_;
    };
}  // namespace ompl::cbf
