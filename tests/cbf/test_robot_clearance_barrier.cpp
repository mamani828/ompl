#define BOOST_TEST_MODULE RobotClearanceBarrierTest
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/cbf/RobotClearanceBarrier.h>
#include <ompl/robots/HolonomicMobileManipulator.h>
#include <ompl/robots/Reachy2.h>
#include <ompl/util/RandomNumbers.h>

namespace sdf = ompl::sdf;
using Reachy2 = ompl::robots::Reachy2;
using MobileReachy2 = ompl::robots::HolonomicMobileManipulator<Reachy2>;

// Both robot flavours the Reachy2-family barrier has to serve: a fixed-base arm
// (nBaseJoints == 0) and the same arm behind a planar holonomic base
// (nBaseJoints == 3). Every templated case below runs once per type.
using RobotTypes = boost::mpl::list<Reachy2, MobileReachy2>;

template <typename Robot>
using Barrier = ompl::cbf::RobotClearanceBarrier<Robot>;

namespace
{
    // A single spherical obstacle: signed distance and gradient are exact in
    // closed form, so the barrier can be checked against the truth.
    const Eigen::Vector3d obstacleCenter(0.5, 0.3, 1.1);
    constexpr double obstacleRadius = 0.15;
    constexpr double voxel = 0.02;
    constexpr double valueTolerance = 1e-3;
    constexpr double rowTolerance = 5e-2;

    sdf::DistanceFn sphereField(const Eigen::Vector3d &center, double radius)
    {
        return [center, radius](const Eigen::Vector3d &p) { return (p - center).norm() - radius; };
    }

    // Wide enough to cover the mobile base's translation range plus the arm's
    // reach from anywhere in it, and the fixed-base arm's reach around its origin.
    Eigen::AlignedBox3d reachableBox()
    {
        return Eigen::AlignedBox3d(Eigen::Vector3d(-2.2, -2.2, -0.5), Eigen::Vector3d(2.2, 2.2, 2.5));
    }

    const sdf::GridSDF &obstacleField()
    {
        static const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), reachableBox(), voxel);
        return field;
    }

    template <typename Robot>
    typename Robot::Configuration midConfiguration()
    {
        return 0.5 * (Robot::lowerBounds() + Robot::upperBounds());
    }

    // Legacy safety clamp `RobotCBFControlFilter::maxSpeed()` applies -- reused
    // here so barrier tests move the robot at physically realistic speeds
    // instead of Reachy2's unclamped 100 rad/s placeholder.
    template <typename Robot>
    typename Robot::Configuration clampedSpeed()
    {
        return Robot::velocityLimits().cwiseMin(Robot::Configuration::Constant(1.2));
    }

    // Sampled away from the periodic base yaw's +-pi wrap boundary, so a small
    // perturbation never straddles the wrap.
    template <typename Robot>
    typename Robot::Configuration randomConfiguration(ompl::RNG &rng)
    {
        typename Robot::Configuration q;
        const auto lo = Robot::lowerBounds();
        const auto hi = Robot::upperBounds();
        for (int j = 0; j < Barrier<Robot>::nJoints; ++j)
        {
            if constexpr (Barrier<Robot>::nBaseJoints >= 3)
                if (j == 2)
                {
                    q[j] = rng.uniformReal(-2.5, 2.5);
                    continue;
                }
            q[j] = rng.uniformReal(lo[j], hi[j]);
        }
        return q;
    }

    // `evaluateScreened()` at an infinite threshold behaves like a full,
    // unscreened evaluation: nothing is skipped, so `active == constraintCount()`
    // and `constraint[row] == row` (identity), matching the ordering every test
    // below assumes.
    template <typename Robot>
    void fullEvaluate(const Barrier<Robot> &barrier, const typename Robot::Configuration &q,
                       typename Barrier<Robot>::Evaluation &out)
    {
        using Values = typename Barrier<Robot>::Values;
        barrier.evaluateScreened(q, Values::Constant(std::numeric_limits<double>::infinity()), out);
    }

    // The sphere no joint moves -- "base_link" for the fixed-base arm, and still
    // true for the mobile wrapper since `spheres()` forwards unchanged from the
    // arm and its influence mask is arm-relative.
    template <typename Robot>
    std::size_t noJointMovesThisSphere()
    {
        const auto &spheres = Robot::spheres();
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            if (spheres[i].influence == 0)
                return i;
        return Robot::nSpheres;
    }
}  // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(ValuesMatchTheAnalyticField, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());

    for (const auto &q : {midConfiguration<Robot>(), Robot::lowerBounds(), Robot::upperBounds()})
    {
        typename Barrier<Robot>::Evaluation evaluation;
        fullEvaluate(barrier, q, evaluation);
        BOOST_REQUIRE(evaluation.inBounds);

        const auto kin = robot.kinematics(q);
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
        {
            const Eigen::Vector3d p = Robot::sphereCenter(kin, i);
            // worldMargin is a compile-time constant on this barrier (unlike UR5's
            // ClearanceBarrier, which the equivalent test constructs with margin=0),
            // so it has to be part of the expected value here.
            const double exact =
                (p - obstacleCenter).norm() - obstacleRadius - Robot::spheres()[i].radius - Barrier<Robot>::worldMargin;
            BOOST_CHECK_LE(std::abs(evaluation.values[static_cast<Eigen::Index>(i)] - exact), valueTolerance);
        }
    }
}

// See test_clearance_barrier.cpp's RowsMatchTheAnalyticGradient for why the
// tolerance is far looser than valueTolerance: GridSDF's gradient is only
// first-order accurate, having differentiated an already-interpolated field.
BOOST_AUTO_TEST_CASE_TEMPLATE(RowsMatchTheAnalyticGradient, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    const auto q = midConfiguration<Robot>();
    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);
    const auto kin = robot.kinematics(q);

    for (std::size_t i = 0; i < Robot::nSpheres; ++i)
    {
        const Eigen::Vector3d p = Robot::sphereCenter(kin, i);
        const Eigen::Vector3d gradient = (p - obstacleCenter).normalized();
        const typename Robot::Configuration exact = Robot::sphereJacobian(kin, i).transpose() * gradient;
        const Eigen::Index row = static_cast<Eigen::Index>(i);
        BOOST_CHECK_LE((evaluation.rows.row(row).transpose() - exact).norm(), rowTolerance);
    }
}

// The sphere no joint can move contributes zero to every arm column; on the
// mobile robot its base columns must still move it (translating/rotating the
// base moves every sphere), so they are generically nonzero.
BOOST_AUTO_TEST_CASE_TEMPLATE(NoJointSphereRowIsZeroOnArmColumnsAndNonzeroOnBaseColumnsWhenABaseExists, Robot,
                              RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    const auto q = midConfiguration<Robot>();
    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);

    const std::size_t sphere = noJointMovesThisSphere<Robot>();
    BOOST_REQUIRE_LT(sphere, Robot::nSpheres);

    constexpr int nBaseJoints = Barrier<Robot>::nBaseJoints;
    constexpr int armJoints = Barrier<Robot>::nJoints - nBaseJoints;
    const auto row = evaluation.rows.row(static_cast<Eigen::Index>(sphere));
    BOOST_CHECK_EQUAL(row.tail(armJoints).norm(), 0.0);
    if constexpr (nBaseJoints > 0)
        BOOST_CHECK_GT(row.head(nBaseJoints).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ClearanceGoesNegativeInsideAnObstacle, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const auto kin = robot.kinematics(q);

    const std::size_t sphere = noJointMovesThisSphere<Robot>() == 0 ? 1 : 0;  // any occupied sphere
    const Eigen::Vector3d occupied = Robot::sphereCenter(kin, sphere);
    const sdf::GridSDF field(sphereField(occupied, 0.1), reachableBox(), voxel);
    const Barrier<Robot> barrier(robot, field, q);

    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);
    BOOST_CHECK(!barrier.safe(q));
    BOOST_CHECK_LT(evaluation.values.minCoeff(), 0.0);
    BOOST_CHECK_LE(evaluation.values[static_cast<Eigen::Index>(sphere)], -0.1);
}

// GridSDF clamps out-of-bounds queries, which *over*-reports clearance -- the
// one direction a barrier must never fail in.
BOOST_AUTO_TEST_CASE_TEMPLATE(LeavingTheFieldIsReported, Robot, RobotTypes)
{
    const Robot robot;
    const Eigen::AlignedBox3d tiny(Eigen::Vector3d(-0.05, -0.05, 0.8), Eigen::Vector3d(0.05, 0.05, 1.0));
    const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), tiny, voxel);
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, field, q);

    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);
    BOOST_CHECK(!evaluation.inBounds);

    const Barrier<Robot> reference(robot, obstacleField(), q);
    typename Barrier<Robot>::Evaluation referenceEvaluation;
    fullEvaluate(reference, q, referenceEvaluation);
    BOOST_CHECK(referenceEvaluation.inBounds);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SafeAgreesWithFullEvaluationOnWorldAndSelf, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());

    ompl::RNG rng;
    for (int sample = 0; sample < 20; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation evaluation;
        fullEvaluate(barrier, q, evaluation);

        double world = 0.0, self = 0.0;
        const bool safe = barrier.safe(q, &world, &self);
        BOOST_CHECK_CLOSE(world, evaluation.values.head(Barrier<Robot>::nSpheres).minCoeff(), 1e-9);
        if (barrier.enabledSelfPairs() > 0)
            BOOST_CHECK_CLOSE(self, evaluation.values.tail(static_cast<Eigen::Index>(barrier.enabledSelfPairs())).minCoeff(),
                              1e-9);
        BOOST_CHECK_EQUAL(safe, evaluation.inBounds && world >= 0.0 && self >= 0.0);
    }
}

// Every self-collision row is the exact analytic derivative of the pair
// clearance -- unlike the world rows, which are only as good as the field's
// interpolated gradient -- so central differences should agree closely.
BOOST_AUTO_TEST_CASE_TEMPLATE(SelfPairRowsMatchCentralDifferences, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    constexpr double step = 1e-6;

    ompl::RNG rng;
    double worst = 0.0;
    for (int sample = 0; sample < 15; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation evaluation;
        fullEvaluate(barrier, q, evaluation);

        for (int j = 0; j < Barrier<Robot>::nJoints; ++j)
        {
            typename Robot::Configuration offset = Robot::Configuration::Zero();
            offset[j] = step;
            typename Barrier<Robot>::Evaluation ahead, behind;
            fullEvaluate(barrier, (q + offset).eval(), ahead);
            fullEvaluate(barrier, (q - offset).eval(), behind);

            for (std::size_t p = 0; p < barrier.enabledSelfPairs(); ++p)
            {
                const Eigen::Index i = Barrier<Robot>::nSpheres + static_cast<Eigen::Index>(p);
                const double finite = (ahead.values[i] - behind.values[i]) / (2.0 * step);
                worst = std::max(worst, std::abs(evaluation.rows(i, j) - finite));
            }
        }
    }
    BOOST_CHECK_LT(worst, 1e-6);
}

// The influence-mask analogue of test_clearance_barrier.cpp's frame-window
// check. Base columns are forced to a literal 0.0 by `pairGradient()` (rigid
// base motion cancels exactly), so that check is exact; an arm joint that
// moves neither or both spheres cancels only algebraically (the dot product
// of a rigid rotation's velocity difference with the separation normal is
// zero, not each term of it), so that one is checked to a tight tolerance
// instead of bit-exactly.
BOOST_AUTO_TEST_CASE_TEMPLATE(SelfPairRowsVanishOutsideTheJointsBetweenTheFramesAndOnEveryBaseColumn, Robot,
                              RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    constexpr int nBaseJoints = Barrier<Robot>::nBaseJoints;
    constexpr int armJoints = Barrier<Robot>::nJoints - nBaseJoints;

    ompl::RNG rng;
    for (int sample = 0; sample < 15; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation evaluation;
        fullEvaluate(barrier, q, evaluation);

        for (std::size_t p = 0; p < barrier.enabledSelfPairs(); ++p)
        {
            const auto pair = Robot::selfPairs()[barrier.selfPairIndex(p)];
            const auto maskA = Robot::spheres()[pair.a].influence;
            const auto maskB = Robot::spheres()[pair.b].influence;
            const Eigen::Index row = Barrier<Robot>::nSpheres + static_cast<Eigen::Index>(p);

            if constexpr (nBaseJoints > 0)
                BOOST_REQUIRE_EQUAL(evaluation.rows.row(row).head(nBaseJoints).norm(), 0.0);

            for (int j = 0; j < armJoints; ++j)
            {
                const bool movesA = (maskA & (1u << j)) != 0;
                const bool movesB = (maskB & (1u << j)) != 0;
                if (movesA == movesB)
                    BOOST_CHECK_LE(std::abs(evaluation.rows(row, nBaseJoints + j)), 1e-9);
            }
        }
    }
}

// The value-level counterpart: rigid base motion moves every sphere together,
// so it changes no self-pair separation at all; an arm joint outside a given
// pair's window (moves neither, or moves both as one rigid subtree) likewise
// cannot change that pair's separation.
BOOST_AUTO_TEST_CASE_TEMPLATE(SelfPairClearanceIgnoresJointsOutsideTheWindowAndAnyBaseMotion, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    constexpr int nBaseJoints = Barrier<Robot>::nBaseJoints;
    constexpr int armJoints = Barrier<Robot>::nJoints - nBaseJoints;

    ompl::RNG rng;
    for (int sample = 0; sample < 10; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation before;
        fullEvaluate(barrier, q, before);

        if constexpr (nBaseJoints > 0)
        {
            for (int j = 0; j < nBaseJoints; ++j)
            {
                auto moved = q;
                moved[j] = (j == 2) ? rng.uniformReal(-2.5, 2.5) : q[j] + rng.uniformReal(-0.3, 0.3);
                typename Barrier<Robot>::Evaluation after;
                fullEvaluate(barrier, moved, after);
                for (std::size_t p = 0; p < barrier.enabledSelfPairs(); ++p)
                {
                    const Eigen::Index row = Barrier<Robot>::nSpheres + static_cast<Eigen::Index>(p);
                    BOOST_CHECK_LE(std::abs(after.values[row] - before.values[row]), 1e-9);
                }
            }
        }

        for (int j = 0; j < armJoints; ++j)
        {
            auto moved = q;
            moved[nBaseJoints + j] =
                rng.uniformReal(Robot::lowerBounds()[nBaseJoints + j], Robot::upperBounds()[nBaseJoints + j]);
            typename Barrier<Robot>::Evaluation after;
            fullEvaluate(barrier, moved, after);

            for (std::size_t p = 0; p < barrier.enabledSelfPairs(); ++p)
            {
                const auto pair = Robot::selfPairs()[barrier.selfPairIndex(p)];
                const auto maskA = Robot::spheres()[pair.a].influence;
                const auto maskB = Robot::spheres()[pair.b].influence;
                if (((maskA & (1u << j)) != 0) == ((maskB & (1u << j)) != 0))
                {
                    const Eigen::Index row = Barrier<Robot>::nSpheres + static_cast<Eigen::Index>(p);
                    BOOST_CHECK_LE(std::abs(after.values[row] - before.values[row]), 1e-9);
                }
            }
        }
    }
}

// The claim that makes row screening sound: clearance cannot fall faster than
// `decreaseRates()` says, for any admissible control -- a global Lipschitz
// bound, not a linearisation, so it must hold over the whole step.
BOOST_AUTO_TEST_CASE_TEMPLATE(DecreaseRatesBoundHowFastClearanceCanActuallyFall, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    const auto maxSpeed = clampedSpeed<Robot>();
    typename Barrier<Robot>::Values rates;
    barrier.decreaseRates(maxSpeed, rates);
    constexpr double duration = 0.05;

    ompl::RNG rng;
    double worstSlack = std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < 300; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation before;
        fullEvaluate(barrier, q, before);

        typename Robot::Configuration u;
        for (int j = 0; j < Barrier<Robot>::nJoints; ++j)
            u[j] = maxSpeed[j] * (rng.uniform01() < 0.5 ? -1.0 : 1.0);

        typename Barrier<Robot>::Evaluation after;
        fullEvaluate(barrier, (q + u * duration).eval(), after);

        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(barrier.constraintCount()); ++i)
        {
            const double allowed = rates[i] * duration;
            const double fell = before.values[i] - after.values[i];
            BOOST_REQUIRE_LE(fell, allowed + 1e-9);
            worstSlack = std::min(worstSlack, allowed - fell);
        }
    }
    BOOST_CHECK_LT(worstSlack, 0.05);
}

// A screened evaluation must agree with a full one on everything it reports.
BOOST_AUTO_TEST_CASE_TEMPLATE(ScreeningIsSelfConsistentAcrossThresholds, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    typename Barrier<Robot>::Values rates;
    barrier.decreaseRates(clampedSpeed<Robot>(), rates);
    const typename Barrier<Robot>::Values threshold = rates * 0.05;

    ompl::RNG rng;
    long kept = 0;
    int evaluations = 0;
    for (int sample = 0; sample < 150; ++sample)
    {
        const auto q = randomConfiguration<Robot>(rng);
        typename Barrier<Robot>::Evaluation full;
        fullEvaluate(barrier, q, full);
        typename Barrier<Robot>::Evaluation screened;
        barrier.evaluateScreened(q, threshold, screened);

        BOOST_REQUIRE_LE((screened.values - full.values).cwiseAbs().maxCoeff(), 1e-12);
        BOOST_REQUIRE_EQUAL(screened.inBounds, full.inBounds);
        BOOST_REQUIRE_LE(screened.active, Barrier<Robot>::maxConstraints);

        for (Eigen::Index r = 0; r < screened.active; ++r)
        {
            const Eigen::Index i = screened.constraint[r];
            BOOST_REQUIRE_LE(screened.values[i], threshold[i]);
            BOOST_REQUIRE_LE((screened.rows.row(r) - full.rows.row(i)).cwiseAbs().maxCoeff(), 1e-12);
        }
        kept += screened.active;
        ++evaluations;
    }
    BOOST_TEST_MESSAGE("mean rows kept: " << static_cast<double>(kept) / evaluations << " of "
                                          << barrier.constraintCount());
}

// The certificate has to hold *along* the span, not merely at its end: sample
// the straight motion it certifies and require the CBF condition every
// constraint would have been held to at every point of it. Run with both a
// zero and a nonzero buffer (RobotClearanceBarrier's superset over the
// fixed-base original's 3-argument certifiedDuration).
BOOST_AUTO_TEST_CASE_TEMPLATE(NothingCanBindWithinTheCertifiedDuration, Robot, RobotTypes)
{
    const Robot robot;
    const Barrier<Robot> barrier(robot, obstacleField(), midConfiguration<Robot>());
    const auto maxSpeed = clampedSpeed<Robot>();
    // The rate `RobotCBFControlFilter` runs this family at. It has to be this brisk for
    // there to be anything to certify: every joint at full speed can spend what clearance
    // these robots have in a few tens of milliseconds, and a certificate is empty
    // whenever a row is already inside `rate/kappa`.
    constexpr double kappa = 30.0;  // 1/s

    ompl::RNG rng;
    for (double buffer : {0.0, 0.02})
    {
        int certified = 0;
        for (int sample = 0; sample < 200; ++sample)
        {
            const auto q = randomConfiguration<Robot>(rng);
            typename Barrier<Robot>::Evaluation evaluation;
            fullEvaluate(barrier, q, evaluation);
            if (!evaluation.inBounds || evaluation.values.minCoeff() <= buffer)
                continue;

            typename Robot::Configuration u;
            for (int j = 0; j < Barrier<Robot>::nJoints; ++j)
                u[j] = maxSpeed[j] * (rng.uniform01() < 0.5 ? -1.0 : 1.0);

            // Empty is a legitimate answer for a rate-based certificate: a row already
            // inside `rate/kappa` has no span over which it provably cannot bind.
            const double duration = barrier.certifiedDuration(evaluation, u, kappa, buffer);
            if (duration <= 0.0)
                continue;
            ++certified;

            constexpr int samples = 15;
            for (int step = 1; step <= samples; ++step)
            {
                const double t = duration * step / samples;
                typename Barrier<Robot>::Evaluation along;
                fullEvaluate(barrier, (q + u * t).eval(), along);
                const double envelope = std::exp(-kappa * t);
                for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(barrier.constraintCount()); ++i)
                    BOOST_REQUIRE_GE(along.values[i],
                                     envelope * (evaluation.values[i] - buffer) + buffer - 1e-9);
            }
        }
        BOOST_REQUIRE_GT(certified, 20);
    }
}

// The box the field was baked over is not an obstacle, so no barrier value
// falls as a centre approaches it -- but a query outside it is clamped and
// comes back optimistic. A certificate that only watched clearances would
// happily run a sphere out of the field; this one is cut short by the
// boundary as well.
BOOST_AUTO_TEST_CASE_TEMPLATE(TheCertificateStopsAtTheEdgeOfTheField, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const auto kin = robot.kinematics(q);

    Eigen::AlignedBox3d box;
    for (std::size_t i = 0; i < Robot::nSpheres; ++i)
        box.extend(Robot::sphereCenter(kin, i));
    box.min().array() -= 0.15;
    box.max().array() += 0.15;

    const sdf::GridSDF cramped(sphereField(Eigen::Vector3d(0.0, 0.0, 40.0), 0.1), box, voxel);
    const Barrier<Robot> barrier(robot, cramped, q);

    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);
    BOOST_REQUIRE(evaluation.inBounds);
    BOOST_REQUIRE_GT(evaluation.values.head(Barrier<Robot>::nSpheres).minCoeff(), 1.0);
    BOOST_REQUIRE_LE(evaluation.boundary.minCoeff(), 0.15);

    typename Robot::Configuration u = Robot::Configuration::Zero();
    u[0] = clampedSpeed<Robot>()[0];
    // A deliberately permissive rate: 1 ms of horizon, so no clearance row can be what
    // binds and the boundary term -- which carries no rate at all -- is isolated.
    constexpr double permissive = 1000.0;
    const double duration = barrier.certifiedDuration(evaluation, u, permissive);
    BOOST_REQUIRE_GT(duration, 0.0);
    BOOST_REQUIRE(std::isfinite(duration));

    const Eigen::AlignedBox3d roomy(Eigen::Vector3d(-3.0, -3.0, -1.0), Eigen::Vector3d(3.0, 3.0, 3.0));
    const sdf::GridSDF wide(sphereField(Eigen::Vector3d(0.0, 0.0, 40.0), 0.1), roomy, 4 * voxel);
    const Barrier<Robot> unbounded(robot, wide, q);
    typename Barrier<Robot>::Evaluation wideEvaluation;
    fullEvaluate(unbounded, q, wideEvaluation);
    BOOST_CHECK_GT(unbounded.certifiedDuration(wideEvaluation, u, permissive), duration);
}
