#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <ompl/robots/RevoluteBounds.h>
#include <ompl/sdf/GridSDF.h>

namespace ompl::cbf
{
    /// The swept-enclosure primitives `tightenLeverBounds()` runs, under a name that
    /// cannot collide with this header's own `detail`.
    namespace detail_sweep = ompl::robots::detail;

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

        /// Does a robot's link step carry the fixed rotation and the joint axis that the
        /// swept-enclosure tightening needs? A minimal model -- `PointCBFGateBenchmark`'s
        /// translation-only `PointRobot` -- describes its links with parent, active joint
        /// and offset alone. That is enough for `buildLeverBounds()`'s triangle
        /// inequality and not enough for `tightenLeverBounds()`, which then leaves the
        /// table exactly as the triangle inequality built it.
        template <typename Step, typename = void>
        struct HasSweepGeometry : std::false_type
        {
        };

        template <typename Step>
        struct HasSweepGeometry<Step, std::void_t<decltype(Step::rotation), decltype(Step::axis)>>
          : std::true_type
        {
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

        /// The margins a barrier gets when the caller does not say. Kept as the values
        /// they were when they were compile-time constants, so every existing caller is
        /// unchanged.
        static constexpr double defaultWorldMargin = 0.010;
        static constexpr double defaultSelfMargin = 0.005;

        /// What the barrier actually holds `h >= 0` against. Per-instance rather than
        /// per-type because a filter that has *replaced* the collision checker has to
        /// over-reserve against the field it reads: `GridSDF` is a trilinear interpolant
        /// of a sampled field, and a shelf board thinner than one voxel is represented
        /// badly enough that a barrier at the audited margin is not a barrier at all. See
        /// `interpolationBuffer()`, and `ClearanceBarrier::guarding()` for the UR5's
        /// version of the same argument.
        double worldMargin() const
        {
            return worldMargin_;
        }

        double selfMargin() const
        {
            return selfMargin_;
        }

        /// How much a filter must over-reserve so that the invariant it enforces survives
        /// the field's interpolation error: one voxel, which is what bounds how far a
        /// trilinear interpolant can sit above the true distance.
        static double interpolationBuffer(const sdf::GridSDF &field)
        {
            return field.spacing().maxCoeff();
        }
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
        RobotClearanceBarrier(const Robot &robot, const sdf::GridSDF &field, const Configuration &reference,
                              double worldMargin = defaultWorldMargin,
                              double selfMargin = defaultSelfMargin)
          : robot_(robot), field_(field), worldMargin_(worldMargin), selfMargin_(selfMargin)
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
                if (a.influence != b.influence && gap > selfMargin_ + 0.02)
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
            Centers gradients;
            out.active = 0;
            out.inBounds = true;
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers.col(static_cast<Eigen::Index>(i)) = Robot::sphereCenter(kin, i);
                out.inBounds = out.inBounds && field_.inBounds(centers.col(static_cast<Eigen::Index>(i)));
                out.boundary[static_cast<Eigen::Index>(i)] = boundaryClearance(centers.col(static_cast<Eigen::Index>(i)));
                const auto query = field_.screenedValueGradient(centers.col(static_cast<Eigen::Index>(i)),
                    Robot::spheres()[i].radius, worldMargin_, threshold[static_cast<Eigen::Index>(i)]);
                out.values[static_cast<Eigen::Index>(i)] = query.value - Robot::spheres()[i].radius - worldMargin_;
                gradients.col(static_cast<Eigen::Index>(i)) = query.gradient;
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const auto pair = Robot::selfPairs()[selfPairs_[p]];
                out.values[static_cast<Eigen::Index>(nSpheres) + static_cast<Eigen::Index>(p)] =
                    (centers.col(pair.a) - centers.col(pair.b)).norm() - Robot::spheres()[pair.a].radius -
                    Robot::spheres()[pair.b].radius - selfMargin_;
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
                    out.rows.row(row) = gradients.col(index).transpose() * Robot::sphereJacobian(kin, flat);
                else
                    out.rows.row(row) = pairGradient(kin, centers, flat - static_cast<std::size_t>(nSpheres)).transpose();
            }
        }

        /// How long the constant control \p control may be applied from the
        /// configuration \p evaluation was taken at before any row could bind,
        /// i.e. the span over which a filter enforcing this barrier is provably
        /// a no-op, for the continuous-time condition `dh/dt >= -kappa h`.
        /// \p buffer is subtracted (floored at zero) from every barrier value
        /// first -- see `RobotCBFControlFilter`'s "integration buffer" doc for
        /// why a caller might want that, and why it defaults to zero (a no-op,
        /// reproducing the original fixed-base behaviour exactly). See
        /// `ClearanceBarrier::certifiedDuration()` for the full Lipschitz
        /// argument this specializes, including why the span honours the
        /// exponential envelope and not merely non-negativity.
        double certifiedDuration(const Evaluation &evaluation, const Configuration &control, double kappa,
                                 double buffer = 0.0) const
        {
            const Configuration speed = control.cwiseAbs();
            const auto worldTravel = (leverBounds_ * speed).eval();
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            const double horizon = kappa > 0.0 ? 1.0 / kappa : std::numeric_limits<double>::infinity();
            double duration = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                if (worldTravel[index] <= 0.0)
                    continue;
                const double allowance =
                    std::min(std::max(evaluation.values[index] - buffer, 0.0) / lipschitz -
                                 worldTravel[index] * horizon,
                             evaluation.boundary[index]);
                duration = std::min(duration, allowance / worldTravel[index]);
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const double travel = pairLeverBounds_.row(static_cast<Eigen::Index>(p)).dot(speed);
                if (travel > 0.0)
                    duration = std::min(duration,
                        std::max(evaluation.values[static_cast<Eigen::Index>(nSpheres) +
                                                    static_cast<Eigen::Index>(p)] - buffer, 0.0) / travel - horizon);
            }
            return std::max(duration, 0.0);
        }

        /// The convex polytope of joint displacements around a configuration that are
        /// certified collision-free, as a per-constraint workspace budget.
        ///
        /// This is `ClearanceBarrier::certifiedRegion()` for the templated barrier and
        /// the argument is identical -- see there for why the displacements satisfying
        ///
        ///     sum_k Lever[i][k]     |dq_k| <= slack_i = min(h_i / L, boundary_i)
        ///     sum_k PairLever[p][k] |dq_k| <= slack_p = h_p
        ///
        /// form a centrally symmetric polytope, and for what the lever arms are
        /// conservative about (they are maxima over all of configuration space, so the
        /// polytope is loose by however far the arm is folded from its worst pose).
        ///
        /// A region built from a screened evaluation is as good as one from a full one:
        /// screening drops *rows*, and this reads only values and boundaries, which
        /// `evaluateScreened()` fills for every constraint regardless.
        struct CertifiedRegion
        {
            /// Per-constraint workspace budget, in metres, floored at zero -- so an
            /// already-violated constraint makes the region degenerate rather than wrong.
            /// Indexed like `Evaluation::values`: sphere i at i, enabled self-pair p at
            /// `nSpheres + p`.
            Values slack;
            /// How many self-pair entries of `slack` are meaningful, matching the
            /// barrier's enabled set. Recorded so a region cannot be read against a
            /// differently-configured barrier.
            int selfPairs{0};
            /// False when the evaluation left the field's box, where no barrier value can
            /// be trusted. `contains()` then certifies nothing.
            bool valid{false};
        };

        /// The certified region around the configuration \p evaluation was taken at.
        CertifiedRegion certifiedRegion(const Evaluation &evaluation) const
        {
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);

            CertifiedRegion region;
            region.slack.setZero();
            region.selfPairs = static_cast<int>(selfPairs_.size());
            region.valid = evaluation.inBounds;
            // Leaving the baked box is bounded by plain workspace travel with no field
            // gradient in it, exactly as in `certifiedDuration()` and for the same
            // reason: a clamped query over-reports clearance, and no barrier value sees
            // it coming.
            region.slack.template head<nSpheres>() =
                (evaluation.values.template head<nSpheres>() / lipschitz)
                    .cwiseMin(evaluation.boundary)
                    .cwiseMax(0.0);
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const Eigen::Index row =
                    static_cast<Eigen::Index>(nSpheres) + static_cast<Eigen::Index>(p);
                region.slack[row] = std::max(evaluation.values[row], 0.0);
            }
            return region;
        }

        /// How long the constant control \p control may be applied from \p region's
        /// configuration before any barrier could reach zero: the *safety* certificate,
        /// against `certifiedDuration()`'s no-op one.
        ///
        /// Longer than `certifiedDuration()` by the `1/kappa` lookahead the no-op claim
        /// costs, and longer again because the slack here is the whole clearance rather
        /// than the clearance less that lookahead. What it gives up is that the motion is
        /// no longer *the* filtered one: `h` stays non-negative over the span, but inside
        /// it `h` may decay faster than the CBF's exponential envelope allows. Forward
        /// invariance survives either way -- the span ends with `h >= 0`, the filter
        /// resumes, and a zero control is always admissible for a single integrator.
        ///
        /// See `ClearanceBarrier::safeDuration()` for the full argument and
        /// `FilteredStateSpace::setSafeHops()` for which certificate a hop may spend.
        double safeDuration(const CertifiedRegion &region, const Configuration &control) const
        {
            if (!region.valid)
                return 0.0;
            const Configuration speed = control.cwiseAbs();
            const auto worldTravel = (leverBounds_ * speed).eval();

            // A minimum of ratios does not need a ratio per row: `slack / travel` can
            // only lower the running best when `slack < travel * best`, which is a
            // multiply and a compare, so a division is paid only where the bound actually
            // improves. Identical, not approximate -- see `ClearanceBarrier::safeScale()`,
            // where the same rewrite measured 0.391 -> 0.290 us over 343 rows.
            double duration = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                const double rate = worldTravel[index];
                if (rate <= 0.0)  // no joint that moves this sphere is moving
                    continue;
                if (region.slack[index] < rate * duration)
                    duration = region.slack[index] / rate;
            }
            const int pairs = std::min(region.selfPairs, static_cast<int>(selfPairs_.size()));
            for (int p = 0; p < pairs; ++p)
            {
                const Eigen::Index row = static_cast<Eigen::Index>(nSpheres) + p;
                const double rate = pairLeverBounds_.row(p).dot(speed);
                if (rate <= 0.0)
                    continue;
                if (region.slack[row] < rate * duration)
                    duration = region.slack[row] / rate;
            }
            return std::max(duration, 0.0);
        }

        /// Is `q + delta` certified collision-free by \p region? One matvec per family,
        /// no kinematics and no field query.
        ///
        /// The region is convex and centred on its own configuration, so a true answer
        /// covers the whole straight segment to `q + delta`, not merely its endpoint.
        bool contains(const CertifiedRegion &region, const Configuration &delta) const
        {
            if (!region.valid)
                return false;
            const Configuration absolute = delta.cwiseAbs();
            if (((leverBounds_ * absolute).array() >
                 region.slack.template head<nSpheres>().array())
                    .any())
                return false;
            const int pairs = std::min(region.selfPairs, static_cast<int>(selfPairs_.size()));
            for (int p = 0; p < pairs; ++p)
                if (pairLeverBounds_.row(p).dot(absolute) >
                    region.slack[static_cast<Eigen::Index>(nSpheres) + p])
                    return false;
            return true;
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
                worstWorld = std::min(worstWorld, field_.distance(center) - Robot::spheres()[i].radius - worldMargin_);
            }
            for (const std::size_t source : selfPairs_)
            {
                const auto pair = Robot::selfPairs()[source];
                worstSelf = std::min(worstSelf, (centers.col(pair.a) - centers.col(pair.b)).norm() -
                                                     Robot::spheres()[pair.a].radius - Robot::spheres()[pair.b].radius -
                                                     selfMargin_);
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
        ///
        /// The arm columns come from the triangle inequality: `|p - o_j|` is at most the
        /// sphere's own offset plus every fixed link offset between it and joint j, which
        /// bounds the perpendicular distance to j's axis over all of configuration space.
        /// `tightenLeverBounds()` then improves on it wherever the swept-enclosure
        /// construction can, and this remains the cap and the fallback.
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
            tightenLeverBounds();
        }

        /// Tighten the arm columns of `leverBounds_` with the complete swept-enclosure
        /// search `robots::UR5::leverArmBounds()` runs, generalised from UR5's serial
        /// chain to the link *tree* these models describe.
        ///
        /// ### What is being tightened, and why the loose bound is loose
        ///
        /// The quantity wanted is `|dp_i/dq_j| = |axis_j x (p_i - o_j)|`, the
        /// perpendicular distance from sphere i's centre to joint j's axis, maximised
        /// over every configuration. `buildLeverBounds()` bounds it by the *full*
        /// distance from the joint origin, which throws away the axial component: a
        /// sphere sitting far along a joint's own axis is charged as if all of that
        /// offset were leverage, when rotating the joint barely moves it at all.
        ///
        /// ### The construction
        ///
        /// Walk from the sphere's link up to the root. A point expressed in link `l`'s
        /// frame maps into its parent's as `R_l (Rot(a_l, q_l) p) + t_l`, so at an active
        /// joint the set of places the sphere can be -- over that joint's whole range --
        /// is a disc about `a_l`, and a disc's radius *is* the perpendicular distance
        /// bound. `detail::sweepBounds()` carries such an enclosure through one link
        /// transform and re-encloses it about the next active axis, which both absorbs
        /// that joint's freedom and reads off its lever arm. Every branch (cylinder,
        /// axial ball, origin ball) is kept, because the smallest radius at one joint
        /// need not stay smallest further up.
        ///
        /// Fixed links carry no freedom, so their transforms are composed into the next
        /// active joint's -- that composition is the only thing UR5's serial version did
        /// not have to do.
        ///
        /// Results are folded in with `min`, so this can only ever tighten; a robot whose
        /// chain this cannot walk keeps the triangle-inequality bound. Base columns are
        /// deliberately untouched.
        ///
        /// `OMPL_ROBOT_LEVER_BOUNDS=arm_length` skips the tightening, for the
        /// process-isolated ablation `scripts/benchmark_lipschitz_bounds.py` runs against
        /// the UR5's equivalent switch. Read once, on first use.
        void tightenLeverBounds()
        {
            using StepType = std::decay_t<decltype(Robot::steps()[0])>;
            if constexpr (!detail::HasSweepGeometry<StepType>::value)
                return;
            else
            {
            static const bool armLengthOnly = []
            {
                const char *mode = std::getenv("OMPL_ROBOT_LEVER_BOUNDS");
                return mode != nullptr && std::string(mode) == "arm_length";
            }();
            if (armLengthOnly)
                return;

            const auto &steps = Robot::steps();
            const auto offsetOf = [&steps](int link)
            {
                const auto &step = steps[static_cast<std::size_t>(link - 1)];
                Eigen::Matrix3d rotation;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        rotation(r, c) = step.rotation[static_cast<std::size_t>(3 * r + c)];
                return std::make_pair(rotation, Eigen::Vector3d(step.xyz.data()));
            };

            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index row = static_cast<Eigen::Index>(i);
                const Eigen::Vector3d center(Robot::spheres()[i].center.data());

                // The fixed transform composed since the enclosure's frame was fixed:
                // identity while walking the links below the first active joint, then
                // reset at every active joint to that joint's own offset.
                Eigen::Matrix3d pendingRotation = Eigen::Matrix3d::Identity();
                Eigen::Vector3d pendingTranslation = Eigen::Vector3d::Zero();
                std::vector<detail_sweep::SweepBound> states;
                bool started = false;

                for (int link = Robot::spheres()[i].link; link > 0;)
                {
                    const auto &step = steps[static_cast<std::size_t>(link - 1)];
                    const auto [rotation, translation] = offsetOf(link);
                    if (step.active >= 0)
                    {
                        const Eigen::Vector3d axis =
                            Eigen::Vector3d(step.axis.data()).normalized();
                        constexpr double pad = 1e-12;
                        if (!started)
                        {
                            // The sphere centre carried up to this joint's frame by the
                            // fixed links below it. A rotation about `axis` preserves
                            // distance to `axis`, so measuring here is measuring in the
                            // joint's own frame.
                            const Eigen::Vector3d point = pendingRotation * center + pendingTranslation;
                            const double radius = detail_sweep::perpendicular(point, axis);
                            states.assign(
                                1, {point.dot(axis) * axis, axis, radius + pad, 0.0, true});
                            leverBounds_(row, nBaseJoints + step.active) =
                                std::min(leverBounds_(row, nBaseJoints + step.active), radius + pad);
                            started = true;
                        }
                        else
                        {
                            states = detail_sweep::sweepBounds(states, pendingRotation,
                                                               pendingTranslation, axis);
                            double best = std::numeric_limits<double>::infinity();
                            for (const auto &state : states)
                                best = std::min(best, state.radius);
                            leverBounds_(row, nBaseJoints + step.active) =
                                std::min(leverBounds_(row, nBaseJoints + step.active), best);
                        }
                        // The enclosure now lives in this joint's frame; its own offset
                        // is what carries it to the parent.
                        pendingRotation = rotation;
                        pendingTranslation = translation;
                    }
                    else
                    {
                        // A fixed link: compose it and carry on, introducing no freedom.
                        pendingTranslation = rotation * pendingTranslation + translation;
                        pendingRotation = rotation * pendingRotation;
                    }
                    link = step.parent;
                }
            }
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
        double worldMargin_{defaultWorldMargin};
        double selfMargin_{defaultSelfMargin};
        std::vector<std::size_t> selfPairs_;
        Eigen::Matrix<double, nSpheres, nJoints> leverBounds_;
        Eigen::Matrix<double, nSelfPairs, nJoints> pairLeverBounds_;
    };
}  // namespace ompl::cbf
