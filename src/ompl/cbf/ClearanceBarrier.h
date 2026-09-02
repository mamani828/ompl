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
        ///
        /// Row-major, which is how it is used at every point: `evaluateScreened()` writes
        /// one whole row per surviving barrier, `CBFControlFilter` dots one whole row
        /// against a candidate control, and qpmad walks the active rows. Column-major put
        /// the six entries of a row 343 doubles apart, so each of those touched six cache
        /// lines instead of one; the solve measures about 14% faster reading a contiguous
        /// block, for identical arithmetic.
        using Rows = Eigen::Matrix<double, nConstraints, nJoints, Eigen::RowMajor>;

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

        /// Slack for the squared-domain comparisons in `evaluateScreened()` and
        /// `certifiedDuration()`, as a relative widening of the squared bound.
        ///
        /// Testing `|p_a - p_b| - offset <= limit` as `dSq <= (offset + limit)^2` is exact
        /// in real arithmetic and loses about half the mantissa in floating point: the
        /// squared form resolves the barrier value only to roughly `offset * eps / 2`,
        /// around 1e-17 m here. That is far below any physical scale in this class -- the
        /// margins are millimetres -- but it is not below the scale at which a *ratio*
        /// like `certifiedDuration()`'s `h / rate` is taken, and there a term can be
        /// dropped that the linear test would have kept.
        ///
        /// So the bound is widened by a few epsilon, which decides that lost resolution in
        /// the conservative direction every time: a pair within rounding of its threshold
        /// is *kept* (one more constraint row than strictly needed) and a pair within
        /// rounding of the certificate's current bound is *charged* (a certificate no
        /// longer than the exact one). Both are the directions that cost work rather than
        /// safety, which is the only asymmetry that matters here.
        static constexpr double squaredCompareSlack = 8.0 * std::numeric_limits<double>::epsilon();

        struct Evaluation
        {
            /// h_i(q). Populated for every constraint by `evaluate()`, but only where it
            /// could matter by `evaluateScreened()` -- see there. A caller wanting all
            /// 343 values unconditionally wants `evaluate()` or `values()`.
            Values values;
            /// `|p_a - p_b|^2` for every self-collision pair, always, and the reason
            /// `values` need not be. Squaring is exact, so a test on a pair's barrier
            /// value can be made here instead, and the square root paid only where the
            /// answer turns out to depend on it.
            Eigen::Matrix<double, nSelfPairs, 1> pairDistanceSq;
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
        /// clearance is at or below \p threshold — normally `decreaseRates(maxSpeed) * dt`.
        ///
        /// The saving is the point: a skipped sphere costs one interpolated *value*, while
        /// an included one costs an interpolated gradient and a Jacobian contraction on
        /// top, and then a row in the QP. In open space almost every sphere is skipped and
        /// this collapses to the cost of a collision check.
        ///
        /// What a caller gives up, and it is a real change rather than an optimisation:
        /// the discrete CBF condition `h(q + u dt) >= (1 - gamma) h(q)` is enforced only
        /// for the spheres that were included. Skipped spheres are guaranteed to stay
        /// **safe** (`h_i > 0`) by the Lipschitz argument above, but not to decay at the
        /// prescribed rate. Safety is the invariant that matters; the decay rate is a
        /// smoothness preference. Audit rather than assume — see `guarding()`.
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

            // When the box encloses the reach, `inBounds` is true and `boundary` is
            // infinite by construction, for every sphere at every configuration -- so the
            // 40 box tests are 40 tests of a theorem. Split the loop rather than branch
            // inside it, since the condition is a property of the field, not of q.
            if (enclosesReach_)
            {
                Robot::sphereCenters(kin, centers);
                out.boundary.setConstant(std::numeric_limits<double>::infinity());
            }
            else
            {
                for (std::size_t i = 0; i < Robot::nSpheres; ++i)
                {
                    const Eigen::Index index = static_cast<Eigen::Index>(i);
                    const Eigen::Vector3d center = Robot::sphereCenter(kin, i);
                    centers.col(index) = center;
                    out.inBounds = out.inBounds && field_.inBounds(center);
                    out.boundary[index] = field_.boundaryClearance(center);
                }
            }

            {
                ScopedTimer timer("sdf_query");
                field_.distanceBatch(centers, distances);
            }

            // Screening is decided here rather than in a pass of its own. The flat index
            // puts every world constraint before every pair, so appending world rows in
            // this loop and pair rows in the next one produces exactly the row order a
            // single pass over `0..nConstraints` did -- and each family now tests its own
            // values while they are still in registers.
            Eigen::Matrix<Eigen::Index, nSpheres, 1> activeWorld;
            Eigen::Index activeWorldCount = 0;

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
                if (h <= threshold[index])
                {
                    out.constraint[out.active++] = static_cast<int>(index);
                    activeWorld[activeWorldCount++] = index;
                }
            }

            // The values-only pass is even cheaper here than for the world: a subtraction
            // and a norm off centres already in hand, with no field query at all.
            //
            // Cheaper still without the norm. A pair's barrier value is only ever
            // *compared* here -- against its screening threshold, and against the
            // smallest value so far -- and both comparisons survive squaring:
            //
            //     |p_a - p_b| - offset <= limit   <=>   dSq <= (offset + limit)^2
            //
            // for `offset + limit >= 0`, which is exact rather than approximate. So the
            // loop keeps squared distances and spends a square root only on a pair that
            // is screened in or that actually lowers the minimum -- a handful, against
            // 303 before. The rest never have a `values` entry written, which is the
            // contract change `Evaluation::values` documents; `pairDistanceSq` is what
            // `certifiedDuration()` reads in their place.
            double smallestPair = std::numeric_limits<double>::infinity();
            {
                ScopedTimer selfTimer("self_collision");
                // Hoisted out of the loop: each of these is a function call returning a
                // static table, and there are 303 pairs, every filter call.
                const auto &pairs = Robot::selfPairs();
                const auto &offsets = Robot::selfPairOffsets();
                for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                {
                    const Eigen::Index pairIndex = static_cast<Eigen::Index>(p);
                    const double distanceSq =
                        (centers.col(static_cast<Eigen::Index>(pairs[p].a)) -
                         centers.col(static_cast<Eigen::Index>(pairs[p].b)))
                            .squaredNorm();
                    out.pairDistanceSq[pairIndex] = distanceSq;

                    const double offset = offsets[pairIndex] + selfMargin_;
                    // The largest barrier value at which this pair still changes an
                    // answer: it either binds within the step, or it is the new worst.
                    const double limit = std::max(threshold[nSpheres + pairIndex], smallestPair);
                    const double bound = offset + limit;
                    // Inclusive, matching `h <= limit`, and widened by
                    // `squaredCompareSlack` so a pair that squaring cannot resolve is
                    // kept rather than dropped.
                    if (bound < 0.0 || distanceSq > bound * bound * (1.0 + squaredCompareSlack))
                        continue;

                    const double h = std::sqrt(distanceSq) - offset;
                    out.values[nSpheres + pairIndex] = h;
                    if (h < smallestPair)
                    {
                        smallestPair = h;
                        out.worstPair = p;
                    }
                    if (h <= threshold[nSpheres + pairIndex])
                    {
                        const Eigen::Index row = out.active++;
                        out.constraint[row] = static_cast<int>(nSpheres + pairIndex);
                        out.rows.row(row) = Robot::selfPairGradient(kin, centers, p).transpose();
                    }
                }
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

                // Row k, not a looked-up row index: the world family was appended first,
                // so the k-th surviving sphere is the k-th row.
                for (Eigen::Index k = 0; k < activeWorldCount; ++k)
                {
                    const std::size_t sphere = static_cast<std::size_t>(activeWorld[k]);
                    out.rows.row(k) =
                        Robot::barrierGradient(kin, sphere, activeGradients.col(k)).transpose();
                }
            }
        }

        /// The same bound read backwards: how long the constant control \p u may be
        /// applied from the configuration \p evaluation was taken at before *any* row
        /// could bind. Zero when one already binds.
        ///
        /// This is the screening argument turned into a step length. A row binds when
        /// the discrete CBF condition `h_i(q + u t) >= (1 - gamma) h_i(q)` stops holding
        /// with room to spare, and `h_i` cannot fall faster than `rate_i`, so the
        /// condition survives for as long as `rate_i * t <= gamma * h_i` — and it
        /// survives at every point of the interval, not merely at its end, because the
        /// same inequality holds for every prefix. Taking the minimum over spheres gives
        /// a duration over which the filter is *provably a no-op*: integrating \p u for
        /// any shorter span yields exactly the motion the filter would have produced,
        /// step by step, at no further cost. That is what makes skipping it sound rather
        /// than merely optimistic.
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
        /// ### How it is computed, which is not how it reads
        ///
        /// Written out, this is a minimum of 343 quotients, and spelling it that way costs
        /// 343 divisions and a 303x6 matrix product on a path that runs once per tree
        /// edge. Neither is necessary, and neither changes the answer:
        ///
        /// - **The quotients become products.** A term only matters if it *beats* the
        ///   running minimum, and `allowance / travel < duration` is `allowance <
        ///   duration * travel` for positive `travel`. So each constraint costs a multiply
        ///   and a compare, and the division is paid only when the minimum actually moves
        ///   — a handful of times, not 343.
        /// - **The pair travel bounds become suffix sums.** `Robot::travelBounds()`
        ///   builds them in 240 products instead of 1818; see
        ///   `Robot::selfPairTravelIndex()` for why one table serves both families.
        /// - **The pair barrier values stay squared.** `Evaluation::values` is only
        ///   populated for the pairs that could matter (see `evaluateScreened()`), so this
        ///   reads `Evaluation::pairDistanceSq` and compares in the squared domain,
        ///   taking a square root only for a pair that improves the bound.
        ///
        /// The result is the same minimum over the same 343 terms.
        double certifiedDuration(const Evaluation &evaluation, const Configuration &u,
                                 double gamma) const
        {
            Robot::TravelBounds travel;
            Robot::travelBounds(u.cwiseAbs(), travel);

            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            const double gammaOverLipschitz = gamma / lipschitz;
            const double inverseGamma = 1.0 / gamma;

            double duration = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < nSpheres; ++i)
            {
                const double rate = travel(i, 0);
                if (rate <= 0.0)  // no joint that moves this sphere is moving
                    continue;
                const double allowance =
                    std::min(gammaOverLipschitz * evaluation.values[i], evaluation.boundary[i]);
                if (allowance < duration * rate)
                    duration = allowance / rate;
            }

            const auto &index = Robot::selfPairTravelIndex();
            const auto &base = Robot::selfPairOffsets();
            for (Eigen::Index p = 0; p < nSelfPairs; ++p)
            {
                // Zero here is common rather than exceptional: only the joints strictly
                // between the two frames can change a pair's separation at all.
                const double rate = travel(index(p, 0), index(p, 1));
                if (rate <= 0.0)
                    continue;

                // h_p = sqrt(dSq) - offset, and h_p * gamma / rate < duration is
                // sqrt(dSq) < offset + duration * rate / gamma, so squaring both sides
                // says whether this pair can lower the bound -- without a square root.
                //
                // The squared test only *selects a candidate*. It is widened by
                // `squaredCompareSlack`, so it admits pairs that turn out not to improve
                // anything, and the update below must therefore be an honest comparison on
                // the exact value rather than an assignment. Getting that wrong is not a
                // rounding matter: a widened test that assigns unconditionally can raise
                // the certificate, which is the unsafe direction.
                const double offset = base[p] + selfMargin_;
                const double bound = offset + duration * rate * inverseGamma;
                if (bound <= 0.0 ||
                    evaluation.pairDistanceSq[p] > bound * bound * (1.0 + squaredCompareSlack))
                    continue;

                const double candidate =
                    gamma * (std::sqrt(evaluation.pairDistanceSq[p]) - offset) / rate;
                if (candidate < duration)
                    duration = candidate;
            }
            return std::max(duration, 0.0);
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
        /// A discrete CBF step certifies h(q + u dt) >= (1-gamma) h(q) using a
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
        /// Measured over 10k rollout waypoints at gamma up to 1.0 and steps up to
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
                const auto &pairs = Robot::selfPairs();
                for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
                {
                    const Eigen::Index index = static_cast<Eigen::Index>(p);
                    const Eigen::Index row = nSpheres + index;
                    const double distanceSq =
                        (centers.col(static_cast<Eigen::Index>(pairs[p].a)) -
                         centers.col(static_cast<Eigen::Index>(pairs[p].b)))
                            .squaredNorm();
                    // Filled here too, so an Evaluation from evaluate() is a valid input
                    // to certifiedDuration(), which reads it rather than `values`.
                    out.pairDistanceSq[index] = distanceSq;
                    const double h = std::sqrt(distanceSq) -
                                     Robot::selfPairOffsets()[index] - selfMargin_;

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

            // Spelled the same way as `evaluate()` and `evaluateScreened()` -- one square
            // root of a squared norm, minus one combined offset -- so that all three
            // agree to the last bit and `worstValue()` really is the minimum of what an
            // Evaluation reports.
            const auto &pairs = Robot::selfPairs();
            const auto &offsets = Robot::selfPairOffsets();
            for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
            {
                const Eigen::Index index = static_cast<Eigen::Index>(p);
                out[nSpheres + index] =
                    std::sqrt((centers.col(static_cast<Eigen::Index>(pairs[p].a)) -
                               centers.col(static_cast<Eigen::Index>(pairs[p].b)))
                                  .squaredNorm()) -
                    offsets[index] - selfMargin_;
            }
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
