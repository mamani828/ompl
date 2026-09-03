#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include <Eigen/Core>

#include <ompl/cbf/Profiler.h>
#include <ompl/robots/UR5.h>
#include <ompl/sdf/GridSDF.h>

namespace ompl::cbf
{
    /// The barrier function a CBF steering step constrains, for a spherized UR5
    /// in a baked workspace signed distance field.
    ///
    /// For each collision sphere i with radius r_i and center p_i(q):
    ///
    ///     h_i(q) = d(p_i(q)) - r_i - margin
    ///
    /// and, by the chain rule through the sphere-center Jacobian,
    ///
    ///     dh_i/dq = (dp_i/dq)^T grad d(p_i)
    ///
    /// `evaluate()` returns both: the values say how much clearance each sphere
    /// has, and the rows are the linear constraint rows a QP steering step needs.
    /// The robot is safe exactly when every h_i >= 0.
    ///
    /// This class is deliberately tied to `robots::UR5` for now. The only things
    /// it asks of a robot are sphere centers, sphere Jacobians, and radii, so it
    /// generalizes to other spherized robots (or to a mobile base contributing
    /// extra Jacobian columns) behind an interface later.
    ///
    /// ### The arm against itself
    ///
    /// A workspace field says nothing about the arm folding through itself, so the
    /// same barrier carries a second family of rows — one per sphere pair in
    /// `Robot::selfPairs()`:
    ///
    ///     h_ab(q) = |p_a(q) - p_b(q)| - r_a - r_b - selfMargin
    ///
    /// These are not a separate object, because they are not a separate concept: a
    /// pair is a sphere whose obstacle happens to be another sphere. It produces one
    /// value, one row, one screening threshold and one certificate term, in the same
    /// units, consumed by the same loops. So both families share one flat index —
    /// `i < nSpheres` is sphere i against the world, `nSpheres + p` is pair p — and
    /// everything downstream, `CBFControlFilter` included, is indifferent to which
    /// kind a row is.
    ///
    /// Two respects in which the pair rows are *better* behaved than the world ones,
    /// both exploited below: the distance between two points is exactly 1-Lipschitz,
    /// where an interpolated field is only approximately so, and a pair has no baked
    /// box it can fall out of.
    ///
    /// One respect in which they are worse, and it is the cost of the feature. No
    /// pair in the table is ever clearer than 58.2 mm, at any configuration, so the
    /// step `certifiedDuration()` can certify is now bounded however empty the
    /// workspace is. The long certified hops that used to cover a whole tree
    /// extension in open space are gone; what remains is a shorter certificate that
    /// still beats stepping.
    ///
    /// ### The margin
    ///
    /// `margin` is what makes the barrier *conservative*, and it has to absorb
    /// three separate errors, none of which are optional:
    ///
    /// - **Sphere under-coverage.** VAMP's sphere model does not enclose the UR5's
    ///   links; mesh vertices sit up to 30.5 mm outside it
    ///   (`scripts/ur5_sphere_coverage.py`). Without this term, h_i > 0 does not
    ///   imply the real robot is clear.
    /// - **SDF discretization.** `GridSDF` interpolates, with error up to about a
    ///   third of a voxel, and it can err optimistically.
    /// - **Step linearization.** A QP using `rows` reasons about a *linear* model
    ///   of h over a finite step; the true h is curved, because the sphere centers
    ///   travel on arcs.
    ///
    /// `defaultMargin` is a single number chosen to cover all three at a 25 mm
    /// voxel and a joint step of ~0.08 rad. It is deliberately blunt — a tighter,
    /// per-sphere bound is derivable from the Jacobian column norms (each one is
    /// the sphere's distance to that joint's axis), but that wants validating
    /// against brute-force rollouts before anything relies on it.
    ///
    /// Note that a *filter* enforcing h >= 0 does not by itself deliver h >= 0: see
    /// `interpolationBuffer()` and `guarding()`, which is what to use when the
    /// filter has replaced the collision checker.
    ///
    /// ### Staying inside the field
    ///
    /// `GridSDF` clamps out-of-bounds queries to the nearest boundary node, which
    /// *over*-reports clearance — the one failure direction a barrier cannot
    /// tolerate. So every evaluation reports whether all sphere centers were
    /// inside the baked box, and a caller must treat `inBounds == false` as "no
    /// usable barrier here" rather than as safety.
    class ClearanceBarrier
    {
    public:
        using Robot = robots::UR5;
        using Configuration = Robot::Configuration;

        static constexpr int nSpheres = static_cast<int>(Robot::nSpheres);
        static constexpr int nSelfPairs = static_cast<int>(Robot::nSelfPairs);
        static constexpr int nJoints = static_cast<int>(Robot::nJoints);

        /// Every barrier this class knows about, world and self, under one flat index:
        /// `i < nSpheres` is sphere i against the field, `nSpheres + p` is
        /// `Robot::selfPairs()[p]`.
        static constexpr int nConstraints = nSpheres + nSelfPairs;

        /// One barrier value per constraint.
        using Values = Eigen::Matrix<double, nConstraints, 1>;
        /// Row i is dh_i/dq — the constraint row barrier i contributes.
        using Rows = Eigen::Matrix<double, nConstraints, nJoints>;

        /// Covers sphere under-coverage (30.5 mm) plus SDF discretization plus
        /// step linearization. See the class comment.
        static constexpr double defaultMargin = 0.06;

        /// The self-collision counterpart, and zero, which wants justifying.
        ///
        /// Each of the three errors `defaultMargin` absorbs either does not arise between
        /// two spheres or is not this constant's job. There is no field to discretize and
        /// no interpolation to disagree with itself. Step linearization is real but tiny
        /// and belongs to `defaultSelfBuffer`, which is what a *filter* over-reserves;
        /// measured, it never exceeded rounding.
        ///
        /// That leaves the one thing a margin here could buy — clearance against the
        /// meshes the spheres stand in for — and it cannot buy it. Covering VAMP's 30.5 mm
        /// of sphere under-coverage on both bodies would need more than 60 mm, which
        /// exceeds the 58.2 mm the tightest pair in the table is *ever* clear by: the
        /// barrier would be infeasible at every configuration. There is no partial version
        /// of that argument either, only a partial cost. MotionBenchMaker's own UR5
        /// endpoints — poses its mesh checker calls collision-free — come within 0.1 mm of
        /// sphere-model contact, so every millimetre asked for here rules out problems the
        /// benchmark considers valid, and 10 mm rules out 58 of the 689.
        ///
        /// So the guarantee is stated exactly and no more: the sphere model does not
        /// self-intersect. That is the quantity the demos' `worstSelfOverlap()` audit
        /// reports, which is what makes the audit able to contradict this class. Callers
        /// wanting physical clearance between links should ask for it explicitly, knowing
        /// it costs reachable configurations.
        static constexpr double defaultSelfMargin = 0.0;

        struct Evaluation
        {
            Values values;                ///< h_i(q), always for every constraint
            /// How far sphere i's centre may travel before leaving the SDF's box, always
            /// for every sphere -- or infinity when the box provably encloses everywhere
            /// the arm can reach, since then no centre can leave it however far it goes.
            /// Only `certifiedDuration()` reads it; see there for why a barrier value
            /// alone does not bound a whole segment of motion. World spheres only: a
            /// pair of sphere centres is not a query against the field and cannot leave
            /// anything.
            Eigen::Matrix<double, nSpheres, 1> boundary;
            Rows rows;                    ///< dh_i/dq -- only the first `active` are filled
            /// Which barrier each of the first `active` rows belongs to, so a caller can
            /// pair row r with `values[constraint[r]]`. The identity for a full evaluation.
            Eigen::Matrix<int, nConstraints, 1> constraint;
            int active{nConstraints};     ///< how many rows of `rows` are meaningful
            /// Index of the smallest *world* h_i, and never a self pair.
            ///
            /// Keeping this world-only is a deliberate restriction rather than an
            /// oversight. Callers read `rows.row(worst)` as "which way is the obstacle"
            /// and steer along it; a self row answers a different question, and its
            /// column 0 is identically zero for every pair inside the arm, so joint 0
            /// would be told it cannot help when in fact it was never asked.
            std::size_t worst{0};
            std::size_t worstPair{0};     ///< pair index of the smallest h_ab, separately
            bool inBounds{true};          ///< were all centers inside the SDF's box?
        };

        /// How fast each sphere's barrier can possibly fall, per unit time, given a
        /// per-joint speed limit: `rate_i = maxGrad * sum_k maxSpeed_k * L[i][k]`.
        ///
        /// This is the Lipschitz bound that makes screening sound. Over a step of duration
        /// dt, `h_i` cannot decrease by more than `rate_i * dt`, so a sphere with
        /// `h_i > rate_i * dt` **cannot** reach zero during the step whatever control is
        /// applied — its constraint row cannot bind and need not be built, let alone
        /// solved against. Both factors are configuration-independent:
        /// `Robot::leverArmBounds()` bounds `|dp_i/dq_k|` over all of configuration space,
        /// and `GridSDF::maxGradientNorm()` bounds the field's gradient over the whole
        /// grid.
        ///
        /// The bound is conservative twice over — the lever arms are not tight, and it
        /// assumes every joint runs at full speed in the worst direction simultaneously —
        /// so it over-selects rows rather than under-selecting them. That is the safe
        /// direction: a loose bound costs work, a tight-but-wrong one costs safety.
        ///
        /// Self-collision pairs get the same treatment from
        /// `Robot::selfPairLeverArms()`, minus the field's Lipschitz constant: the
        /// distance between two sphere centres is 1-Lipschitz exactly, so there is
        /// nothing to multiply by.
        Values decreaseRates(const Configuration &maxSpeed) const
        {
            const Configuration speed = maxSpeed.cwiseAbs();
            Values rates;
            rates.head<nSpheres>() = field_.maxGradientNorm() * (Robot::leverArmBounds() * speed);
            rates.tail<nSelfPairs>() = Robot::selfPairLeverArms() * speed;
            return rates;
        }

        /// Barrier values for every sphere, but constraint rows only for the spheres whose
        /// clearance is at or below \p threshold — normally
        /// `decreaseRates(maxSpeed) * max(dt, 1/kappa)`.
        ///
        /// The saving is the point: a skipped sphere costs one interpolated *value*, while
        /// an included one costs an interpolated gradient and a Jacobian contraction on
        /// top, and then a row in the QP. In open space almost every sphere is skipped and
        /// this collapses to the cost of a collision check.
        ///
        /// Under the continuous-time condition `dh_i/dq u >= -kappa h_i` the threshold
        /// buys back what the discrete-time version had to give up. A row is satisfied by
        /// *every* admissible control once `rate_i <= kappa h_i`, because `rate_i` bounds
        /// `|dh_i/dq u|` over the whole control box; such a row cannot change the feasible
        /// set, so dropping it leaves the QP's solution bit-for-bit identical rather than
        /// merely safe. Screening at `rate_i / kappa` is therefore an optimisation and
        /// nothing more. The `dt` half of the maximum is a separate guarantee for a
        /// separate question — a dropped row must also stay non-negative for the span the
        /// caller is about to integrate, which needs `h_i > rate_i dt` — and taking the
        /// larger of the two horizons buys both at once.
        void evaluateScreened(const Configuration &q, const Values &threshold, Evaluation &out) const
        {
            const Robot::Kinematics kin = robot_.kinematics(q);

            out.inBounds = true;
            out.worst = 0;
            out.worstPair = 0;
            out.active = 0;

            // Centres are kept because the survivors need them again for the gradient
            // query, and because every pair value is a subtraction between two of them.
            // Forward kinematics is not worth repeating.
            Robot::SphereCenters centers;
            Eigen::Matrix<double, nSpheres, 1> distances;

            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                const Eigen::Vector3d center = Robot::sphereCenter(kin, i);
                centers.col(index) = center;
                out.inBounds = out.inBounds && field_.inBounds(center);
                out.boundary[index] = boundaryClearance(center);
            }

            {
                ScopedTimer timer("sdf_query");
                field_.distanceBatch(centers, distances);
            }

            double smallest = std::numeric_limits<double>::infinity();
            const auto &allSpheres = Robot::spheres();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                const double h = distances[index] - allSpheres[i].radius - margin_;
                out.values[index] = h;
                if (h < smallest)
                {
                    smallest = h;
                    out.worst = i;
                }
            }

            // The values-only pass is even cheaper here than for the world: a subtraction
            // and a norm off centres already in hand, with no field query at all.
            //
            // This loop is scalar on purpose. Rewritten structure-of-arrays -- gather the
            // pair endpoints into contiguous per-axis vectors and let the square roots
            // vectorise -- it was measurably *slower*, 1273 ns against 1032 ns for the
            // whole evaluation. The square root is not what costs; the gather is, and the
            // array form pays it anyway and then spends 7 KB of intermediates that this
            // version keeps in registers.
            double smallestPair = std::numeric_limits<double>::infinity();
            {
                ScopedTimer selfTimer("self_collision");
                // Hoisted out of the loop: each of these is a function call returning a
                // static table, and there are 303 pairs, every filter call.
                const auto &pairs = Robot::selfPairs();
                const auto &radii = Robot::selfPairRadii();
                const auto &margins = Robot::selfPairMargins();
                for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                {
                    const Eigen::Index pairIndex = static_cast<Eigen::Index>(p);
                    const Eigen::Index index = nSpheres + pairIndex;
                    const double h = (centers.col(static_cast<Eigen::Index>(pairs[p].a)) -
                                      centers.col(static_cast<Eigen::Index>(pairs[p].b)))
                                         .norm() -
                                     radii[pairIndex] - margins[pairIndex] - selfMargin_;
                    out.values[index] = h;
                    if (h < smallestPair)
                    {
                        smallestPair = h;
                        out.worstPair = p;
                    }
                }
            }

            // One screening pass over both families: they are the same question asked of
            // different geometry, and the flat index keeps the QP's row order stable.
            Eigen::Matrix<Eigen::Index, nSpheres, 1> activeWorld;
            Eigen::Matrix<Eigen::Index, nSpheres, 1> activeWorldRows;
            Eigen::Index activeWorldCount = 0;

            for (Eigen::Index i = 0; i < nConstraints; ++i)
            {
                if (out.values[i] > threshold[i])
                    continue;

                const Eigen::Index row = out.active++;
                out.constraint[row] = static_cast<int>(i);
                if (i < nSpheres)
                {
                    activeWorld[activeWorldCount] = i;
                    activeWorldRows[activeWorldCount] = row;
                    ++activeWorldCount;
                }
                else
                    out.rows.row(row) =
                        Robot::selfPairGradient(kin, centers, static_cast<std::size_t>(i - nSpheres))
                            .transpose();
            }

            if (activeWorldCount > 0)
            {
                // Fixed capacity (nSpheres is the worst case, every sphere active), so
                // this is stack space, not a heap allocation -- unlike Matrix3Xd/VectorXd,
                // which are Dynamic-sized and would malloc here on every filter call.
                Eigen::Matrix<double, 3, nSpheres> activeCentersBuf;
                Eigen::Matrix<double, nSpheres, 1> activeDistancesBuf;
                Eigen::Matrix<double, 3, nSpheres> activeGradientsBuf;

                auto activeCenters = activeCentersBuf.leftCols(activeWorldCount);
                auto activeDistances = activeDistancesBuf.head(activeWorldCount);
                auto activeGradients = activeGradientsBuf.leftCols(activeWorldCount);

                for (Eigen::Index k = 0; k < activeWorldCount; ++k)
                    activeCenters.col(k) = centers.col(activeWorld[k]);

                {
                    ScopedTimer timer("sdf_query");
                    field_.valueGradientBatch(activeCenters, activeDistances, activeGradients);
                }
                (void)activeDistances;

                for (Eigen::Index k = 0; k < activeWorldCount; ++k)
                {
                    const std::size_t sphere = static_cast<std::size_t>(activeWorld[k]);
                    out.rows.row(activeWorldRows[k]) =
                        Robot::barrierGradient(kin, sphere, activeGradients.col(k)).transpose();
                }
            }
        }

        /// The same bound read backwards: how long the constant control \p u may be
        /// applied from the configuration \p evaluation was taken at before *any* row
        /// could bind. Zero when one already binds.
        ///
        /// This is the screening argument turned into a step length, and under the
        /// continuous-time condition it is an interval statement in its own right rather
        /// than a chain of per-step ones. Row i is satisfied by \p u at time s whenever
        /// `L travel_i <= kappa h_i(s)`, and `h_i(s) >= h_i - L travel_i s`, so the row
        /// cannot bind for as long as
        ///
        ///     t <= h_i / (L travel_i) - 1 / kappa
        ///
        /// where `travel_i` is the decrease rate of *this* control rather than of the
        /// whole box. The bound holds at every point of the interval, not merely at its
        /// end, because the right-hand side was derived from the worst case over the
        /// prefix. Taking the minimum over rows gives a duration over which the filter is
        /// *provably a no-op*: integrating \p u for any shorter span yields exactly the
        /// motion the filter would have produced, step by step, at no further cost. That
        /// is what makes skipping it sound rather than merely optimistic.
        ///
        /// The span also honours the decay the filter promises, which is the stronger
        /// claim and the one worth checking. Over it, `h_i(t) >= h_i e^{-kappa t}`: at
        /// `x = kappa h_i / (L travel_i) >= 1` the linear lower bound at the endpoint is
        /// `L travel_i / kappa` against an exponential `(L travel_i / kappa) x e^{1-x}`,
        /// and `x e^{1-x} <= 1` everywhere, with the interior covered because the gap
        /// between the two rises and then falls and so is smallest at an endpoint.
        ///
        /// The `1 / kappa` subtraction is why a certificate shortens as the decay rate
        /// falls: a slower promised decay binds sooner, not later.
        ///
        /// Two details keep it honest:
        ///
        /// - **The rate is per-control.** `decreaseRates(maxSpeed)` assumes every joint
        ///   runs flat out in the worst direction; asking about the control actually
        ///   applied is the same expression with \p u in its place, and it is several
        ///   times longer.
        /// - **Leaving the box is not falling clearance.** A centre that exits the SDF's
        ///   bounds gets a clamped, over-optimistic distance back, and no barrier value
        ///   sees it coming. So the excursion is bounded by `Evaluation::boundary` too,
        ///   at the plain workspace travel rate (no field gradient involved).
        ///
        /// The Lipschitz constant is floored at 1 rather than taken raw: a true signed
        /// distance field is 1-Lipschitz, `maxGradientNorm()` only ever exceeds that
        /// because of interpolation, and a field with no obstacle in the box reports
        /// zero, which would otherwise certify an unbounded step off the back of a
        /// division by zero.
        /// Self-collision pairs enter the same minimum, and they must: a certificate
        /// taken over the world rows alone would happily run a constant control for a
        /// span during which the arm closes on itself, precisely because no world row
        /// can see that coming. They are charged less than the world rows, though —
        /// no Lipschitz constant, since a distance between two points has one exactly,
        /// and no boundary term, since a pair is not a query against the field.
        double certifiedDuration(const Evaluation &evaluation, const Configuration &u,
                                 double kappa) const
        {
            const Configuration speed = u.cwiseAbs();
            const auto travel = (Robot::leverArmBounds() * speed).eval();
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            // kappa == 0 is "never let clearance fall at all", which no motion towards an
            // obstacle can certify; the reciprocal is infinite and every term goes to zero.
            const double horizon = kappa > 0.0 ? 1.0 / kappa : std::numeric_limits<double>::infinity();

            double duration = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < nSpheres; ++i)
            {
                if (travel[i] <= 0.0)  // no joint that moves this sphere is moving
                    continue;
                // h_i / (L travel_i) - 1/kappa, kept in the allowance/travel shape the
                // boundary term wants: leaving the box is bounded by plain travel, with
                // no field gradient and no decay rate in it.
                const double allowance = std::min(evaluation.values[i] / lipschitz - travel[i] * horizon,
                                                  evaluation.boundary[i]);
                duration = std::min(duration, allowance / travel[i]);
            }

            const auto pairTravel = (Robot::selfPairLeverArms() * speed).eval();
            for (Eigen::Index p = 0; p < nSelfPairs; ++p)
            {
                // Zero here is common rather than exceptional: only the joints strictly
                // between the two frames can change a pair's separation at all.
                if (pairTravel[p] <= 0.0)
                    continue;
                duration = std::min(duration,
                                    evaluation.values[nSpheres + p] / pairTravel[p] - horizon);
            }
            return std::max(duration, 0.0);
        }

        /// The combined lever-arm table: row i bounds how far sphere i's centre moves
        /// per radian of joint k, and row `nSpheres + p` bounds how fast pair p's
        /// separation can change. Exactly the two tables `decreaseRates()` multiplies,
        /// stacked under the same flat index everything else here uses, so a region test
        /// is one matvec rather than two loops.
        static const Eigen::Matrix<double, nConstraints, nJoints> &leverArms()
        {
            static const Eigen::Matrix<double, nConstraints, nJoints> table = []
            {
                Eigen::Matrix<double, nConstraints, nJoints> combined;
                combined.topRows<nSpheres>() = Robot::leverArmBounds();
                combined.bottomRows<nSelfPairs>() = Robot::selfPairLeverArms();
                return combined;
            }();
            return table;
        }

        /// The set of joint displacements from the configuration an `Evaluation` was
        /// taken at that are certified collision-free with no further evaluation.
        ///
        /// ### The same table, fed a displacement instead of a velocity
        ///
        /// `decreaseRates()` is named for how it is used, not for what it is. Take the
        /// velocity out of it and what is left is a Lipschitz map from a joint-space
        /// *displacement* to a bound on how far each barrier can fall:
        ///
        ///     h_i(q + dq)  >= h_i(q)  - L * sum_k Lever[i][k]     |dq_k|
        ///     h_ab(q + dq) >= h_ab(q) -     sum_k PairLever[p][k] |dq_k|
        ///
        /// The sums bound how far the relevant centres can move -- each lever arm is a
        /// sphere's greatest distance from that joint's axis, so turning the joint by
        /// `dq_k` moves the centre by at most the product, and the chain composes
        /// additively -- and the field is `L`-Lipschitz on top of that. Feed
        /// `maxSpeed * dt` back in and `decreaseRates() * dt` comes out exactly. The
        /// rate reading is the special case, not the general one, and it is the only
        /// one the class used before this.
        ///
        /// ### What the displacement bound certifies
        ///
        /// Every barrier non-negative means the robot is clear, since `margin` has
        /// already absorbed sphere under-coverage and the field's discretization. So the
        /// displacements that keep all of them non-negative are the ones satisfying
        ///
        ///     sum_k Lever[i][k]     |dq_k| <= slack_i = min(h_i / L, boundary_i)
        ///     sum_k PairLever[p][k] |dq_k| <= slack_p = h_ab
        ///
        /// an intersection of weighted L1 constraints, and therefore a centrally
        /// symmetric convex polytope around q. `slack` is what this returns.
        ///
        /// `certifiedDuration()` is that polytope intersected with the ray `dq = u t`,
        /// less the `1/kappa` its stronger no-op claim costs. Every evaluation has
        /// always contained the region; the ray is what was being kept from it.
        ///
        /// ### Why the polytope rather than a ball
        ///
        /// `contains()` is one `nConstraints x nJoints` matvec against `|dq|`, which is
        /// the same matvec `certifiedDuration()` already performs. The region therefore
        /// costs no more to test than the ray does, and there is no reason to shrink it
        /// to an inscribed ball first. `certifiedRadius()` does shrink it, for the
        /// callers that need a scalar rather than a test.
        ///
        /// ### What it is conservative about
        ///
        /// The lever arms are maxima over the whole configuration space, so the polytope
        /// is loose by however much the arm is folded away from its worst pose. How loose
        /// is a question for measurement, not for this comment.
        ///
        /// `L` is `maxGradientNorm()` floored at one, matching `certifiedDuration()`.
        /// True distance is exactly 1-Lipschitz and the interpolation error is already
        /// inside `margin`, so `L = 1` is defensible and would widen every world
        /// constraint by that factor -- but it spends margin budget nothing else here
        /// spends, and wants auditing before it is taken.
        ///
        /// A region built from a screened evaluation is as good as one from a full
        /// evaluation. Screening drops *rows*; this reads only values and boundaries,
        /// which `evaluateScreened()` fills for every constraint.
        struct CertifiedRegion
        {
            /// Per-constraint workspace budget, in metres. Floored at zero, so a
            /// constraint that is already violated makes the region degenerate rather
            /// than making it wrong.
            Values slack;
            /// False when the evaluation was out of the field's box, where no barrier
            /// value can be trusted. `contains()` then certifies nothing.
            bool valid{false};
        };

        /// The certified region around the configuration \p evaluation was taken at.
        CertifiedRegion certifiedRegion(const Evaluation &evaluation) const
        {
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);

            CertifiedRegion region;
            region.valid = evaluation.inBounds;
            // Leaving the baked box is bounded by plain workspace travel, with no field
            // gradient in it, exactly as in certifiedDuration() -- and for the same
            // reason: a clamped query over-reports clearance, and no barrier value sees
            // it coming.
            region.slack.head<nSpheres>() =
                (evaluation.values.head<nSpheres>() / lipschitz).cwiseMin(evaluation.boundary);
            region.slack.tail<nSelfPairs>() = evaluation.values.tail<nSelfPairs>();
            region.slack = region.slack.cwiseMax(0.0);
            return region;
        }

        /// The certified region at \p q directly, without an `Evaluation`.
        ///
        /// This is the form a caller replacing filter calls actually wants, and it is
        /// cheaper than either evaluation path. The region reads values, boundaries and
        /// in-bounds-ness and nothing else, so there is no gradient query, no Jacobian
        /// contraction and no constraint row -- which is where a barrier evaluation's
        /// time goes, at roughly three times the QP it feeds. `evaluate()` would build
        /// 343 rows this discards; `evaluateScreened()` would build the survivors.
        ///
        /// So the region is not merely longer-reaching than the duration certificate, it
        /// is cheaper to obtain than the call that produces one.
        CertifiedRegion certifiedRegion(const Configuration &q) const
        {
            const Robot::Kinematics kin = robot_.kinematics(q);

            Robot::SphereCenters centers;
            Eigen::Matrix<double, nSpheres, 1> distances;
            Eigen::Matrix<double, nSpheres, 1> boundary;

            CertifiedRegion region;
            region.valid = true;
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                const Eigen::Vector3d center = Robot::sphereCenter(kin, i);
                centers.col(index) = center;
                region.valid = region.valid && field_.inBounds(center);
                boundary[index] = boundaryClearance(center);
            }

            {
                ScopedTimer timer("sdf_query");
                field_.distanceBatch(centers, distances);
            }

            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            const auto &allSpheres = Robot::spheres();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(i);
                const double h = distances[index] - allSpheres[i].radius - margin_;
                region.slack[index] = std::max(std::min(h / lipschitz, boundary[index]), 0.0);
            }

            {
                ScopedTimer selfTimer("self_collision");
                const auto &margins = Robot::selfPairMargins();
                for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                {
                    const Eigen::Index pair = static_cast<Eigen::Index>(p);
                    region.slack[nSpheres + pair] =
                        std::max(Robot::selfPairClearance(centers, p) - margins[pair] - selfMargin_,
                                 0.0);
                }
            }
            return region;
        }

        /// Is `q + delta` certified collision-free by \p region? One matvec, no
        /// kinematics and no field query.
        ///
        /// The region is convex and centred on its own configuration, so a true answer
        /// covers the whole straight segment to `q + delta` and not merely its endpoint.
        static bool contains(const CertifiedRegion &region, const Configuration &delta)
        {
            if (!region.valid)
                return false;
            return ((leverArms() * delta.cwiseAbs()).array() <= region.slack.array()).all();
        }

        /// How long the constant control \p u may be run from \p region's configuration
        /// before any barrier could reach zero. The safety certificate, against
        /// `certifiedDuration()`'s no-op one.
        ///
        /// This is `certifiedRegion()` restricted to the ray `dq = u t`, and it is
        /// exactly `certifiedDuration()` without the `1/kappa`. The two answer different
        /// questions and a caller has to know which it is asking:
        ///
        /// - `certifiedDuration()`: over this span the filter is provably a *no-op*, so
        ///   integrating \p u reproduces the filtered motion exactly.
        /// - `safeDuration()`: over this span `h >= 0`, so integrating \p u is *safe*.
        ///   It is longer -- by `1/kappa` on every row -- but the motion is no longer
        ///   the one the filter would have produced, and `h` may decay faster inside it
        ///   than the exponential envelope allows.
        ///
        /// Forward invariance survives either way: the span ends with `h >= 0`, the
        /// filter resumes, and `u = 0` is always admissible for a single integrator, so
        /// there is no state the longer span can strand the robot in. What is given up
        /// is the envelope, not the safe set. See `noOpTraversalTime()` for the third
        /// option, which keeps the envelope by slowing the traversal down instead.
        static double safeDuration(const CertifiedRegion &region, const Configuration &u)
        {
            if (!region.valid)
                return 0.0;
            const auto travel = (leverArms() * u.cwiseAbs()).eval();

            double duration = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < nConstraints; ++i)
                if (travel[i] > 0.0)  // no joint that moves this constraint is moving
                    duration = std::min(duration, region.slack[i] / travel[i]);
            return duration;
        }

        /// The smallest decay rate at which \p region certifies that \p u satisfies every
        /// CBF row: the least `kappa` for which `(dh_i/dq) u >= -kappa h_i` provably holds
        /// at the configuration \p region was taken at, for every constraint at once.
        ///
        /// ### Why the region can answer a question about gradients
        ///
        /// The row asks about `(dh_i/dq) u`, which is a gradient. The region never
        /// computes one. It does not have to: the same lever-arm table that bounds how
        /// fast a barrier can fall bounds the row's left-hand side from below,
        ///
        ///     (dh_i/dq)  u >= -L travel_i     for a world sphere,
        ///     (dh_ab/dq) u >=   -travel_p     for a self-collision pair,
        ///
        /// with `travel = leverArms() * |u|` as everywhere else here. So row i holds at
        /// any `kappa >= L travel_i / h_i`, and since `slack_i <= h_i / L` for a world row
        /// and `slack_p = h_ab` for a pair, `kappa >= travel_i / slack_i` suffices for
        /// both. The maximum over rows is what this returns, and it is exactly the
        /// reciprocal of `safeDuration()`:
        ///
        ///     requiredGain(region, u) = max_i travel_i / slack_i = 1 / safeDuration(region, u)
        ///
        /// which is the reading worth keeping: **the gain a control needs is the
        /// reciprocal of how long it could be run for.** A control with half a second of
        /// clear road needs 2 /s; one with ten seconds needs 0.1 /s.
        ///
        /// ### What it is for
        ///
        /// It is the second stage of the lexicographic gain program. The first stage --
        /// minimise the deviation from the nominal control -- is the ordinary CBF-QP run
        /// at the *cap*, because for `h_i > 0` a control is feasible for some
        /// `kappa <= kappaMax` exactly when it is feasible at `kappaMax`. That leaves the
        /// gain itself to be read off after the fact rather than optimised, and this is
        /// how it is read off. `CBFControlFilter::Diagnostics::requiredGain` is this
        /// quantity for the control that filter returned.
        ///
        /// ### What it is conservative about
        ///
        /// Everything the region is, and one thing more. The lever arms are maxima over
        /// the whole configuration space, so `travel_i` overstates the true directional
        /// derivative by however much the arm is folded away from its worst pose; the
        /// answer is therefore an upper bound on the gain the gradients would have
        /// demanded, never an under-estimate, which is the safe direction for a number a
        /// certificate is built on. In exchange it sees one thing the gradients cannot:
        /// `slack_i` carries `Evaluation::boundary`, so a control that would walk a sphere
        /// centre out of the baked field -- where clearance is clamped and over-reported,
        /// and no barrier value sees it coming -- is charged for it here.
        ///
        /// Infinite when nothing certifies \p u: the region is invalid, or a row it must
        /// clear has no slack left. Zero when \p u moves no constraint at all, which is
        /// correct rather than degenerate -- a control that changes no clearance needs no
        /// allowance to spend it.
        static double requiredGain(const CertifiedRegion &region, const Configuration &u)
        {
            const double safe = safeDuration(region, u);
            return safe > 0.0 ? 1.0 / safe : std::numeric_limits<double>::infinity();
        }

        /// The largest L-infinity ball inscribed in \p region: every configuration
        /// within this many radians of the centre, on every joint at once, is certified.
        ///
        /// Strictly weaker than `contains()`, and offered only for callers that need a
        /// single number -- a sampler's rejection radius, a nearest-neighbour cutoff.
        static double certifiedRadius(const CertifiedRegion &region)
        {
            if (!region.valid)
                return 0.0;
            static const Values reach = leverArms().rowwise().sum();

            double radius = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < nConstraints; ++i)
                if (reach[i] > 0.0)
                    radius = std::min(radius, region.slack[i] / reach[i]);
            return radius;
        }

        /// How slowly the straight segment \p delta must be traversed for a filter
        /// enforcing `dh/dt >= -kappa h` to be a *no-op* along all of it. Infinite when
        /// \p delta is not strictly inside \p region, where no traversal time suffices.
        ///
        /// This is the part of the story `certifiedDuration()` cannot tell, because it
        /// is asked about a control and so has already had the speed decided for it.
        /// Traverse `delta` over a duration T and the applied control is `delta / T`, so
        /// row i's decrease rate is `travel_i / T` and it fails to bind anywhere on the
        /// segment as long as
        ///
        ///     travel_i / T <= kappa (slack_i - travel_i)
        ///
        /// using `h_i` at its worst point on the segment, which the displacement bound
        /// gives as `slack_i - travel_i`. Solving for T and taking the maximum over rows
        /// gives this. As `T -> infinity` the condition degenerates to `travel_i <
        /// slack_i`, which is exactly membership of the region.
        ///
        /// So `1/kappa` never restricts *where* a filtered edge may go, only how fast it
        /// may be run: every edge strictly inside the region is a no-op edge at some
        /// finite traversal time. That matters because `FilteredStateSpace` is a
        /// geometric space, and a geometric planner does not choose execution speed --
        /// which means the certificate `certifiedDuration()` reports is charging the
        /// planner for a decision the planner never made.
        static double noOpTraversalTime(const CertifiedRegion &region, const Configuration &delta,
                                        double kappa)
        {
            constexpr double never = std::numeric_limits<double>::infinity();
            // kappa == 0 forbids any decay at all, which no motion towards an obstacle
            // can satisfy however slowly it is run.
            if (!region.valid || !(kappa > 0.0))
                return never;

            const auto travel = (leverArms() * delta.cwiseAbs()).eval();
            double time = 0.0;
            for (Eigen::Index i = 0; i < nConstraints; ++i)
            {
                if (travel[i] <= 0.0)  // no joint that moves this constraint is moving
                    continue;
                const double room = region.slack[i] - travel[i];
                if (room <= 0.0)
                    return never;
                time = std::max(time, travel[i] / (kappa * room));
            }
            return time;
        }

        /// Both duration certificates for the control \p u, in one pass.
        ///
        /// `certifiedDuration()` and `safeDuration()` ask the same question of the same
        /// data and differ only in whether a row must stay *inactive* or merely
        /// non-negative. Computing them separately costs two `nConstraints x nJoints`
        /// matvecs and two passes over 343 rows, which the profile put at 45% of a filter
        /// call -- more than the barrier evaluation that produces the numbers. This does
        /// the matvec once and both minima in one sweep.
        ///
        /// The other half of the saving is not dividing. A minimum of ratios does not
        /// need a ratio per row: `slack_i / travel_i` can only lower the running best
        /// when `slack_i < travel_i * best`, which is a multiply and a compare, so a
        /// division is paid only when the bound actually improves -- a handful of times
        /// over 343 rows rather than 343 times. The result is identical, not approximate.
        ///
        /// \p safe is the longer, weaker span (`h >= 0`); \p noOp the shorter one over
        /// which the filter provably would not have acted. `safe >= noOp` always, since
        /// every row's `noOp` term is its `safe` term less the `1/kappa` lookahead.
        void durations(const Evaluation &evaluation, const Configuration &u, double kappa,
                       double &safe, double &noOp) const
        {
            const auto travel = (leverArms() * u.cwiseAbs()).eval();
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            // kappa == 0 forbids any decay, which no motion towards an obstacle can
            // certify as a no-op; the safety span is unaffected by it.
            const double horizon = kappa > 0.0 ? 1.0 / kappa : std::numeric_limits<double>::infinity();

            double safeBest = std::numeric_limits<double>::infinity();
            double noOpBest = std::numeric_limits<double>::infinity();

            for (Eigen::Index i = 0; i < nSpheres; ++i)
            {
                const double rate = travel[i];
                if (rate <= 0.0)  // no joint that moves this sphere is moving
                    continue;
                const double clearance = evaluation.values[i] / lipschitz;
                const double bound = evaluation.boundary[i];

                // Leaving the baked box is bounded by plain travel, with no field
                // gradient and no decay rate in it, so it enters both spans the same way.
                const double safeRoom = std::min(clearance, bound);
                if (safeRoom < rate * safeBest)
                    safeBest = safeRoom / rate;

                const double noOpRoom = std::min(clearance - rate * horizon, bound);
                if (noOpRoom < rate * noOpBest)
                    noOpBest = noOpRoom / rate;
            }

            for (Eigen::Index p = 0; p < nSelfPairs; ++p)
            {
                const Eigen::Index index = nSpheres + p;
                const double rate = travel[index];
                // Zero is common rather than exceptional here: only the joints strictly
                // between two frames can change a pair's separation at all.
                if (rate <= 0.0)
                    continue;
                const double clearance = evaluation.values[index];

                if (clearance < rate * safeBest)
                    safeBest = clearance / rate;
                // The pair's no-op term is `clearance/rate - horizon`, and it improves on
                // the running best exactly when `clearance < rate * (best + horizon)`.
                if (clearance < rate * (noOpBest + horizon))
                    noOpBest = clearance / rate - horizon;
            }

            safe = std::max(safeBest, 0.0);
            noOp = std::max(noOpBest, 0.0);
        }

        /// Neither \p robot nor \p field is copied; both must outlive this object.
        ClearanceBarrier(const Robot &robot, const sdf::GridSDF &field, double margin = defaultMargin,
                         double selfMargin = defaultSelfMargin)
          : robot_(robot)
          , field_(field)
          , margin_(margin)
          , selfMargin_(selfMargin)
          , enclosesReach_(field.bounds().contains(Robot::reachableBounds()))
        {
        }

        /// How much a *filter* must over-reserve so that the invariant it enforces
        /// still holds when checked afterwards.
        ///
        /// A CBF step certifies h(q + u dt) >= h(q) e^{-kappa dt} using a
        /// linear model built from `rows`. `GridSDF` now differentiates the same
        /// trilinear scalar field it evaluates, so value/gradient inconsistency is not
        /// the cause of the remaining error. The finite joint-space step still crosses
        /// a nonlinear robot kinematic map and may cross voxel boundaries where the
        /// trilinear field's gradient jumps. A filter riding h = 0 can therefore land
        /// slightly below it unless the enforced barrier reserves some room.
        ///
        /// So one voxel. This matters specifically when a CBF filter replaces the
        /// collision checker: with a checker in the loop those steps were quietly
        /// truncated as invalid, and nobody noticed.
        static double interpolationBuffer(const sdf::GridSDF &field)
        {
            return field.spacing().maxCoeff();
        }

        /// The same over-reservation for the self-collision rows, and far smaller.
        ///
        /// A sphere pair queries no field and therefore cannot cross a voxel boundary.
        /// Its value and row come from the same two centres, so the only way a step can
        /// land below what the row predicted is the linearization itself — the centres
        /// travel on arcs, and the row is a chord.
        ///
        /// That error is small and, unlike the field's, does shrink with the step.
        /// Measured over 10k rollout waypoints at decay rates up to 1/dt and steps up to
        /// 0.08 rad, the enforced self barrier never fell below its margin by more than
        /// rounding. A millimetre is therefore generous, and generosity is affordable
        /// here in a way it is not for the margin itself: the buffer comes out of the
        /// 58.2 mm the tightest pair in the table ever has, and a millimetre of that is
        /// nothing, where the 60 mm a mesh-level margin would need is everything.
        static constexpr double defaultSelfBuffer = 0.001;

        /// The barrier a *filter* should enforce if you intend to hold the robot to
        /// \p margin. Buffered by `interpolationBuffer()`, so auditing the resulting
        /// motion against a plain `ClearanceBarrier(robot, field, margin)` passes.
        ///
        /// Use this whenever the filter is the only thing standing between the
        /// planner and the obstacles.
        static ClearanceBarrier guarding(const Robot &robot, const sdf::GridSDF &field,
                                         double margin = defaultMargin)
        {
            return guarding(robot, field, margin, interpolationBuffer(field));
        }

        /// As above with the buffer chosen by hand. Smaller is cheaper — the buffer
        /// is clearance the planner cannot use, which is what decides whether a tight
        /// passage stays solvable — but a buffer below what the field and step size
        /// actually require silently reintroduces violations. Audit before trusting
        /// any hand-picked value: interpolate a solution path and evaluate the
        /// unbuffered barrier at every step.
        ///
        /// \p buffer applies to the world margin and \p selfBuffer to the self-collision
        /// one; they are separate because they absorb unrelated errors, and differ by
        /// more than an order of magnitude. See `defaultSelfBuffer`.
        static ClearanceBarrier guarding(const Robot &robot, const sdf::GridSDF &field, double margin,
                                         double buffer, double selfMargin = defaultSelfMargin,
                                         double selfBuffer = defaultSelfBuffer)
        {
            return ClearanceBarrier(robot, field, margin + buffer, selfMargin + selfBuffer);
        }

        /// Barrier values and constraint rows at \p q.
        void evaluate(const Configuration &q, Evaluation &out) const
        {
            const Robot::Kinematics kin = robot_.kinematics(q);

            out.inBounds = true;
            out.worst = 0;
            out.worstPair = 0;
            out.active = nConstraints;

            Robot::SphereCenters centers;
            Eigen::Matrix<double, nSpheres, 1> distances;
            Eigen::Matrix<double, 3, nSpheres> gradients;

            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Vector3d center = Robot::sphereCenter(kin, i);
                const Eigen::Index row = static_cast<Eigen::Index>(i);
                centers.col(row) = center;
                out.inBounds = out.inBounds && field_.inBounds(center);
                out.constraint[row] = static_cast<int>(i);
                out.boundary[row] = boundaryClearance(center);
            }

            {
                ScopedTimer timer("sdf_query");
                field_.valueGradientBatch(centers, distances, gradients);
            }

            double smallest = std::numeric_limits<double>::infinity();
            const auto &allSpheres = Robot::spheres();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Index row = static_cast<Eigen::Index>(i);
                const double h = distances[row] - allSpheres[i].radius - margin_;

                out.values[row] = h;
                out.rows.row(row) = Robot::barrierGradient(kin, i, gradients.col(row)).transpose();

                if (h < smallest)
                {
                    smallest = h;
                    out.worst = i;
                }
            }

            double smallestPair = std::numeric_limits<double>::infinity();
            {
                ScopedTimer selfTimer("self_collision");
                for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                {
                    const Eigen::Index row = nSpheres + static_cast<Eigen::Index>(p);
                    const double h = Robot::selfPairClearance(centers, p) -
                                     Robot::selfPairMargins()[static_cast<Eigen::Index>(p)] -
                                     selfMargin_;

                    out.constraint[row] = static_cast<int>(row);
                    out.values[row] = h;
                    out.rows.row(row) = Robot::selfPairGradient(kin, centers, p).transpose();

                    if (h < smallestPair)
                    {
                        smallestPair = h;
                        out.worstPair = p;
                    }
                }
            }
        }

        Evaluation evaluate(const Configuration &q) const
        {
            Evaluation out;
            evaluate(q, out);
            return out;
        }

        /// Barrier values only. Skips the Jacobians, so this is the cheap call for
        /// "is this configuration safe?" — no constraint rows are produced.
        void values(const Configuration &q, Values &out, bool *inBounds = nullptr) const
        {
            const Robot::Kinematics kin = robot_.kinematics(q);
            if (inBounds != nullptr)
                *inBounds = true;

            Robot::SphereCenters centers;
            Eigen::Matrix<double, nSpheres, 1> distances;
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                const Eigen::Vector3d center = Robot::sphereCenter(kin, i);
                centers.col(static_cast<Eigen::Index>(i)) = center;
                if (inBounds != nullptr)
                    *inBounds = *inBounds && field_.inBounds(center);
            }

            field_.distanceBatch(centers, distances);

            const auto &allSpheres = Robot::spheres();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
                out[static_cast<Eigen::Index>(i)] =
                    distances[static_cast<Eigen::Index>(i)] - allSpheres[i].radius - margin_;

            for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                out[nSpheres + static_cast<Eigen::Index>(p)] =
                    Robot::selfPairClearance(centers, p) -
                    Robot::selfPairMargins()[static_cast<Eigen::Index>(p)] - selfMargin_;
        }

        Values values(const Configuration &q) const
        {
            Values out;
            values(q, out);
            return out;
        }

        /// Tightest clearance over every barrier, world and self. Safe iff this is >= 0.
        double worstValue(const Configuration &q) const
        {
            return values(q).minCoeff();
        }

        bool isSafe(const Configuration &q) const
        {
            return worstValue(q) >= 0.0;
        }

        double margin() const
        {
            return margin_;
        }

        void setMargin(double margin)
        {
            margin_ = margin;
        }

        double selfMargin() const
        {
            return selfMargin_;
        }

        void setSelfMargin(double selfMargin)
        {
            selfMargin_ = selfMargin;
        }

        const Robot &robot() const
        {
            return robot_;
        }

        const sdf::GridSDF &field() const
        {
            return field_;
        }

    private:
        /// How far a centre at \p p may move before it could leave the field, or infinity
        /// when the question cannot arise.
        ///
        /// `Robot::reachableBounds()` encloses every sphere at every reachable
        /// configuration, so a field baked over at least that much can never be queried
        /// outside itself no matter what the arm does -- and charging a certificate for a
        /// boundary it cannot reach would gut it, since the box is drawn tight around the
        /// arm's reach and an extended arm sits right against it.
        double boundaryClearance(const Eigen::Vector3d &p) const
        {
            return enclosesReach_ ? std::numeric_limits<double>::infinity()
                                  : field_.boundaryClearance(p);
        }

        const Robot &robot_;
        const sdf::GridSDF &field_;
        double margin_;
        double selfMargin_;
        bool enclosesReach_;
    };
}  // namespace ompl::cbf
