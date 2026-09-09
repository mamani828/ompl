#define BOOST_TEST_MODULE RobotCBFControlFilterTest
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <qpmad/solver.h>

#include <ompl/cbf/RobotCBFControlFilter.h>
#include <ompl/robots/HolonomicMobileManipulator.h>
#include <ompl/robots/Reachy2.h>
#include <ompl/util/RandomNumbers.h>

namespace sdf = ompl::sdf;
using Reachy2 = ompl::robots::Reachy2;
using MobileReachy2 = ompl::robots::HolonomicMobileManipulator<Reachy2>;
using RobotTypes = boost::mpl::list<Reachy2, MobileReachy2>;

template <typename Robot>
using Barrier = ompl::cbf::RobotClearanceBarrier<Robot>;
template <typename Robot>
using Filter = ompl::cbf::RobotCBFControlFilter<Robot>;
template <typename Robot>
using Status = typename ompl::cbf::RobotControlFilter<Robot>::Status;

namespace
{
    const Eigen::Vector3d obstacleCenter(0.5, 0.3, 1.1);
    constexpr double obstacleRadius = 0.15;
    constexpr double voxel = 0.02;
    // The step this filter family actually integrates at (see the Reachy2 demos), which
    // is what `Filter::kappa` is the rate for.
    constexpr double duration = 0.02;

    sdf::DistanceFn sphereField(const Eigen::Vector3d &center, double radius)
    {
        return [center, radius](const Eigen::Vector3d &p) { return (p - center).norm() - radius; };
    }

    Eigen::AlignedBox3d reachableBox()
    {
        return Eigen::AlignedBox3d(Eigen::Vector3d(-2.2, -2.2, -0.5), Eigen::Vector3d(2.2, 2.2, 2.5));
    }

    /// An obstacle close enough to bind at the mid configuration.
    const sdf::GridSDF &nearField()
    {
        static const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), reachableBox(), voxel);
        return field;
    }

    /// Far enough from every reachable sphere on either robot that no world
    /// clearance row can ever bind.
    const sdf::GridSDF &farField()
    {
        static const sdf::GridSDF field(sphereField(Eigen::Vector3d(10.0, 10.0, 10.0), 0.2), reachableBox(), voxel);
        return field;
    }

    template <typename Robot>
    typename Robot::Configuration midConfiguration()
    {
        return 0.5 * (Robot::lowerBounds() + Robot::upperBounds());
    }

    template <typename Robot>
    typename Robot::Configuration clampedSpeed()
    {
        return Robot::velocityLimits().cwiseMin(Robot::Configuration::Constant(1.2));
    }

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

    template <typename Robot>
    void fullEvaluate(const Barrier<Robot> &barrier, const typename Robot::Configuration &q,
                       typename Barrier<Robot>::Evaluation &out)
    {
        using Values = typename Barrier<Robot>::Values;
        barrier.evaluateScreened(q, Values::Constant(std::numeric_limits<double>::infinity()), out);
    }

    template <typename Robot>
    std::size_t noJointMovesThisSphere()
    {
        const auto &spheres = Robot::spheres();
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            if (spheres[i].influence == 0)
                return i;
        return Robot::nSpheres;
    }

    /// A nominal control aimed straight at the tightest world sphere, scaled by
    /// \p speed.
    template <typename Robot>
    typename Robot::Configuration towardWorstSphere(const Barrier<Robot> &barrier, const typename Robot::Configuration &q,
                                                    double speed)
    {
        typename Barrier<Robot>::Evaluation evaluation;
        fullEvaluate(barrier, q, evaluation);
        Eigen::Index worst = 0;
        evaluation.values.head(Barrier<Robot>::nSpheres).minCoeff(&worst);
        return -evaluation.rows.row(worst).transpose().normalized() * speed;
    }

    // ------------------------------------------------------------------
    // Numerical-equivalence oracles: transcribed verbatim from the
    // pre-refactor ReachyFilter/MobileFilter::filter() bodies (demos/
    // Reachy2CBFPlanningDemo.cpp and Reachy2MobileCBFPlanningDemo.cpp),
    // copied before those classes were deleted. They call the NEW
    // RobotClearanceBarrier -- already separately verified in
    // test_robot_clearance_barrier.cpp to reproduce both originals' barrier
    // math exactly -- so what these oracles isolate is only the piece that
    // changed: the QP objective, the position-bound tiering, and the
    // integration buffer.
    // ------------------------------------------------------------------

    namespace oracle
    {
        // Verbatim port of ReachyFilter::filter (fixed-base, identity Hessian,
        // no integration buffer, no periodic-joint skip).
        Reachy2::Configuration reachyFilter(const Barrier<Reachy2> &barrier, const Reachy2::Configuration &q,
                                            const Reachy2::Configuration &nominal, double dt)
        {
            using RBarrier = Barrier<Reachy2>;
            constexpr int N = RBarrier::nJoints;
            constexpr double kappa = 30.0;
            const Reachy2::Configuration maxSpeed =
                Reachy2::velocityLimits().cwiseMin(Reachy2::Configuration::Constant(1.2));

            RBarrier::Values decreaseRates;
            barrier.decreaseRates(maxSpeed, decreaseRates);
            const RBarrier::Values threshold = decreaseRates * std::max(dt, 1.0 / kappa);

            RBarrier::Evaluation evaluation;
            barrier.evaluateScreened(q, threshold, evaluation);

            Reachy2::Configuration lower = -maxSpeed;
            Reachy2::Configuration upper = maxSpeed;
            const Reachy2::Configuration qlo = Reachy2::lowerBounds();
            const Reachy2::Configuration qhi = Reachy2::upperBounds();
            for (int j = 0; j < N; ++j)
            {
                lower[j] = std::max(lower[j], (qlo[j] - q[j]) / dt);
                upper[j] = std::min(upper[j], (qhi[j] - q[j]) / dt);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }

            const Eigen::Index active = evaluation.active;
            Reachy2::Configuration filtered;
            if (active == 0)
                filtered = nominal.cwiseMax(lower).cwiseMin(upper);
            else
            {
                using Solver = qpmad::SolverTemplate<double, N, 1, RBarrier::maxConstraints>;
                Eigen::Matrix<double, N, N> hessian = Eigen::Matrix<double, N, N>::Identity();
                Reachy2::Configuration objective = -nominal;
                RBarrier::Values rowLower, rowUpper;
                rowUpper.setConstant(std::numeric_limits<double>::infinity());
                for (Eigen::Index row = 0; row < active; ++row)
                    rowLower[row] = -kappa * evaluation.values[evaluation.constraint[row]];
                Solver solver;
                solver.solve(filtered, hessian, objective, lower, upper, evaluation.rows.topRows(active),
                             rowLower.head(active), rowUpper.head(active));
            }
            return filtered;
        }

        // Verbatim port of MobileFilter::filter (speed-weighted Hessian, 1 mm
        // integration buffer, base yaw at column 2 skipped in position bounds).
        MobileReachy2::Configuration mobileFilter(const Barrier<MobileReachy2> &barrier,
                                                  const MobileReachy2::Configuration &q,
                                                  const MobileReachy2::Configuration &nominal, double dt,
                                                  const MobileReachy2::Configuration &lowerPosition,
                                                  const MobileReachy2::Configuration &upperPosition)
        {
            using RBarrier = Barrier<MobileReachy2>;
            constexpr int N = RBarrier::nJoints;
            constexpr double kappa = 30.0;
            constexpr double integrationBuffer = 0.001;
            const MobileReachy2::Configuration maxSpeed = MobileReachy2::velocityLimits();
            const MobileReachy2::Configuration inverseSquared =
                maxSpeed.cwiseInverse().cwiseProduct(maxSpeed.cwiseInverse());

            RBarrier::Values decreaseRates;
            barrier.decreaseRates(maxSpeed, decreaseRates);
            RBarrier::Values threshold = decreaseRates * std::max(dt, 1.0 / kappa);
            threshold.array() += integrationBuffer;

            RBarrier::Evaluation evaluation;
            barrier.evaluateScreened(q, threshold, evaluation);

            MobileReachy2::Configuration lower = -maxSpeed;
            MobileReachy2::Configuration upper = maxSpeed;
            for (int j = 0; j < N; ++j)
            {
                if (j == 2)
                    continue;
                lower[j] = std::max(lower[j], (lowerPosition[j] - q[j]) / dt);
                upper[j] = std::min(upper[j], (upperPosition[j] - q[j]) / dt);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }

            const Eigen::Index active = evaluation.active;
            MobileReachy2::Configuration filtered;
            if (active == 0)
                filtered = nominal.cwiseMax(lower).cwiseMin(upper);
            else
            {
                using Solver = qpmad::SolverTemplate<double, N, 1, RBarrier::maxConstraints>;
                Eigen::Matrix<double, N, N> hessian = Eigen::Matrix<double, N, N>::Zero();
                hessian.diagonal() = inverseSquared;
                MobileReachy2::Configuration objective = -inverseSquared.cwiseProduct(nominal);
                RBarrier::Values rowLower, rowUpper;
                rowUpper.setConstant(std::numeric_limits<double>::infinity());
                for (Eigen::Index row = 0; row < active; ++row)
                    rowLower[row] = -kappa * (evaluation.values[evaluation.constraint[row]] - integrationBuffer);
                Solver solver;
                solver.solve(filtered, hessian, objective, lower, upper, evaluation.rows.topRows(active),
                             rowLower.head(active), rowUpper.head(active));
            }
            return filtered;
        }
    }  // namespace oracle
}  // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(SafeNominalPassesThroughUntouched, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, farField(), q);
    BOOST_REQUIRE(barrier.safe(q));
    const Filter<Robot> filter(barrier);

    const typename Robot::Configuration nominal = Robot::Configuration::Zero();
    typename Robot::Configuration filtered;
    BOOST_CHECK(filter.filter(q, nominal, duration, filtered) == Status<Robot>::Unchanged);
    BOOST_CHECK_EQUAL((filtered - nominal).norm(), 0.0);
}

// Every row must hold, not just the one that bound -- checked against a
// full (unscreened) evaluation of the same barrier the filter used.
BOOST_AUTO_TEST_CASE_TEMPLATE(EveryClearanceRowIsSatisfied, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, nearField(), q);
    const Filter<Robot> filter(barrier);

    typename Barrier<Robot>::Evaluation evaluation;
    fullEvaluate(barrier, q, evaluation);

    typename Robot::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, towardWorstSphere(barrier, q, 10.0), duration, filtered) == Status<Robot>::Filtered);

    const auto slack = evaluation.rows * filtered + evaluation.values * Filter<Robot>::kappa;
    BOOST_CHECK_GE(slack.minCoeff(), -1e-6);
}

// The point of the whole exercise: a nominal that would drive the robot into the
// obstacle is replaced by one spending clearance no faster than `kappa`, and the *true*
// clearance -- not just the linearized prediction -- holds up. A step of constant control
// realizes the first-order term of the exponential envelope, `(1 - kappa dt) h`, which is
// the weaker of the two and so the right floor to assert.
BOOST_AUTO_TEST_CASE_TEMPLATE(ClearanceDecayIsCappedByKappa, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, nearField(), q);
    const Filter<Robot> filter(barrier);

    double beforeWorld = 0.0, beforeSelf = 0.0;
    BOOST_REQUIRE(barrier.safe(q, &beforeWorld, &beforeSelf));
    const double before = std::min(beforeWorld, beforeSelf);

    // Fast enough that the *unfiltered* step still ends in collision over this short a
    // span; the filtered control is clamped to `maxSpeed` either way, so the speed here
    // sets up the comparison rather than changing its outcome.
    const auto nominal = towardWorstSphere(barrier, q, 30.0);
    typename Robot::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, nominal, duration, filtered) == Status<Robot>::Filtered);

    double unfilteredWorld = 0.0, unfilteredSelf = 0.0;
    barrier.safe((q + nominal * duration).eval(), &unfilteredWorld, &unfilteredSelf);
    BOOST_CHECK_LT(std::min(unfilteredWorld, unfilteredSelf), 0.0);  // unfiltered ends in collision

    double afterWorld = 0.0, afterSelf = 0.0;
    barrier.safe((q + filtered * duration).eval(), &afterWorld, &afterSelf);
    const double after = std::min(afterWorld, afterSelf);
    BOOST_CHECK_GE(after, (1.0 - Filter<Robot>::kappa * duration) * before - 1e-3);
    BOOST_CHECK_GT(after, 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SpeedBoxIsRespected, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, farField(), q);
    const Filter<Robot> filter(barrier);
    const auto maxSpeed = Filter<Robot>::maxSpeed();

    typename Robot::Configuration filtered;
    BOOST_CHECK(filter.filter(q, Robot::Configuration::Constant(50.0), duration, filtered) == Status<Robot>::Filtered);
    for (int j = 0; j < Barrier<Robot>::nJoints; ++j)
        BOOST_CHECK_LE(std::abs(filtered[j]), maxSpeed[j] + 1e-9);
}

// With screening and no surviving barrier rows, the QP reduces exactly to a
// component-wise clamp against the speed box.
BOOST_AUTO_TEST_CASE_TEMPLATE(ZeroActiveRowsUseTheExactBoxProjection, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, farField(), q);
    typename Barrier<Robot>::Values rates;
    barrier.decreaseRates(clampedSpeed<Robot>(), rates);
    const typename Barrier<Robot>::Values threshold = rates * std::max(duration, 1.0 / Filter<Robot>::kappa);
    typename Barrier<Robot>::Evaluation evaluation;
    barrier.evaluateScreened(q, threshold, evaluation);
    BOOST_REQUIRE_EQUAL(evaluation.active, 0);

    const Filter<Robot> filter(barrier);
    const auto maxSpeed = Filter<Robot>::maxSpeed();

    typename Robot::Configuration filtered;
    const typename Robot::Configuration inside = Robot::Configuration::Constant(0.01);
    BOOST_CHECK(filter.filter(q, inside, duration, filtered) == Status<Robot>::Unchanged);
    BOOST_CHECK_EQUAL((filtered - inside).norm(), 0.0);

    const typename Robot::Configuration outside = maxSpeed * 5.0;
    const typename Robot::Configuration expected = outside.cwiseMax(-maxSpeed).cwiseMin(maxSpeed);
    BOOST_CHECK(filter.filter(q, outside, duration, filtered) == Status<Robot>::Filtered);
    BOOST_CHECK_LE((filtered - expected).cwiseAbs().maxCoeff(), 1e-9);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(JointLimitsAreRespected, Robot, RobotTypes)
{
    const Robot robot;
    const auto q0 = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, farField(), q0);
    const Filter<Robot> filter(barrier);

    // Start a hair below the upper limit of the first arm joint and push hard.
    constexpr int j = Barrier<Robot>::nBaseJoints;
    auto q = q0;
    q[j] = Robot::upperBounds()[j] - 0.001;

    typename Robot::Configuration nominal = Robot::Configuration::Zero();
    nominal[j] = 10.0;
    typename Robot::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, nominal, duration, filtered) != Status<Robot>::Blocked);
    BOOST_CHECK_LE(filtered[j], 0.001 / duration + 1e-9);
}

// Outside the baked field, GridSDF clamps and over-reports clearance, so
// there is no trustworthy barrier -- that must read as "cannot move".
BOOST_AUTO_TEST_CASE_TEMPLATE(LeavingTheFieldBlocksTheStep, Robot, RobotTypes)
{
    const Robot robot;
    const Eigen::AlignedBox3d tiny(Eigen::Vector3d(-0.05, -0.05, 0.8), Eigen::Vector3d(0.05, 0.05, 1.0));
    const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), tiny, voxel);
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, field, q);
    const Filter<Robot> filter(barrier);

    typename Robot::Configuration filtered;
    BOOST_CHECK(filter.filter(q, Robot::Configuration::Constant(0.1), duration, filtered) == Status<Robot>::Blocked);
    BOOST_CHECK_EQUAL(filtered.norm(), 0.0);
}

// A sphere no joint can move, swallowed by an obstacle, makes its row the
// zero vector while its value is deeply negative: the CBF inequality becomes
// "0 >= a positive number", infeasible for any control -- on the fixed-base
// arm, where nothing else can rescue that sphere. On the mobile robot the
// same sphere's *base* columns are still nonzero (the base can always try to
// back away), so infeasibility isn't guaranteed there; only the "never
// throws" half of the claim is checked for both.
BOOST_AUTO_TEST_CASE_TEMPLATE(CorneredReportsBlockedRatherThanThrowing, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const auto kin = robot.kinematics(q);
    const std::size_t sphere = noJointMovesThisSphere<Robot>();
    BOOST_REQUIRE_LT(sphere, Robot::nSpheres);
    const Eigen::Vector3d center = Robot::sphereCenter(kin, sphere);
    const sdf::GridSDF field(sphereField(center, 0.6), reachableBox(), voxel);
    const Barrier<Robot> barrier(robot, field, q);
    const Filter<Robot> filter(barrier);

    typename Robot::Configuration filtered;
    BOOST_CHECK_NO_THROW(filter.filter(q, Robot::Configuration::Constant(0.1), duration, filtered));
    const auto status = filter.filter(q, Robot::Configuration::Constant(0.1), duration, filtered);
    if constexpr (Barrier<Robot>::nBaseJoints == 0)
    {
        BOOST_CHECK(status == Status<Robot>::Blocked);
        BOOST_CHECK_EQUAL(filtered.norm(), 0.0);
    }
}

// PathControl::check() re-propagates an edge and compares against the stored
// state, so a non-deterministic filter would make every CBF path fail
// validation.
BOOST_AUTO_TEST_CASE_TEMPLATE(FilterIsDeterministic, Robot, RobotTypes)
{
    const Robot robot;
    const auto q = midConfiguration<Robot>();
    const Barrier<Robot> barrier(robot, nearField(), q);
    const Filter<Robot> filter(barrier);
    const auto nominal = towardWorstSphere(barrier, q, 10.0);

    typename Robot::Configuration first, second;
    filter.filter(q, nominal, duration, first);
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        filter.filter(q, nominal, duration, second);
        BOOST_CHECK_EQUAL((first - second).norm(), 0.0);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PassthroughFilterIsIdentity, Robot, RobotTypes)
{
    const ompl::cbf::RobotPassthroughFilter<Robot> filter;
    const typename Robot::Configuration nominal = Robot::Configuration::Constant(123.0);
    typename Robot::Configuration filtered;
    BOOST_CHECK(filter.filter(Robot::Configuration::Zero(), nominal, duration, filtered) == Status<Robot>::Unchanged);
    BOOST_CHECK_EQUAL((filtered - nominal).norm(), 0.0);
}

// An ordinary joint pinned tightly by the position box must be clamped toward
// the remaining room; the periodic base yaw, pinned the same way, must not be
// -- a fixed position limit is meaningless for a wrapping joint.
BOOST_AUTO_TEST_CASE_TEMPLATE(PositionBoundsRespectNonPeriodicJointsAndSkipTheBaseYaw, Robot, RobotTypes)
{
    const Robot robot;
    const auto fullLower = Robot::lowerBounds();
    const auto fullUpper = Robot::upperBounds();

    {
        const auto q = midConfiguration<Robot>();
        const Barrier<Robot> barrier(robot, farField(), q);
        auto lower = fullLower, upper = fullUpper;
        constexpr int j = Barrier<Robot>::nBaseJoints;  // first arm joint
        lower[j] = q[j] - 0.002;
        upper[j] = q[j] + 0.002;
        const Filter<Robot> filter(barrier, lower, upper);

        typename Robot::Configuration nominal = Robot::Configuration::Zero();
        nominal[j] = 10.0;
        typename Robot::Configuration filtered;
        filter.filter(q, nominal, duration, filtered);
        BOOST_CHECK_LE(filtered[j], 0.002 / duration + 1e-9);
    }

    if constexpr (Barrier<Robot>::nBaseJoints >= 3)
    {
        auto q = midConfiguration<Robot>();
        q[2] = 3.0;  // near the +pi wrap
        const Barrier<Robot> barrier(robot, farField(), q);
        auto lower = fullLower, upper = fullUpper;
        lower[2] = q[2];
        upper[2] = q[2];
        const Filter<Robot> filter(barrier, lower, upper);

        typename Robot::Configuration nominal = Robot::Configuration::Zero();
        nominal[2] = 0.4;
        typename Robot::Configuration filtered;
        filter.filter(q, nominal, duration, filtered);
        BOOST_CHECK_GT(std::abs(filtered[2]), 0.1);
    }
}

// The safety net proving the QP-objective unification (identity vs.
// speed-weighted Hessian) and the buffer/yaw-skip generalization didn't
// silently change either demo's behaviour.
BOOST_AUTO_TEST_CASE(NumericalEquivalenceWithOldReachyFilter)
{
    const Reachy2 robot;
    const auto q = midConfiguration<Reachy2>();
    const Barrier<Reachy2> barrier(robot, nearField(), q);
    const Filter<Reachy2> filter(barrier);

    for (const auto &nominal : {Reachy2::Configuration::Zero().eval(), towardWorstSphere(barrier, q, 2.0),
                                towardWorstSphere(barrier, q, 10.0), Reachy2::Configuration::Constant(50.0).eval(),
                                Reachy2::Configuration::Constant(-1.5).eval()})
    {
        Reachy2::Configuration filtered;
        filter.filter(q, nominal, duration, filtered);
        const Reachy2::Configuration expected = oracle::reachyFilter(barrier, q, nominal, duration);
        BOOST_CHECK_LE((filtered - expected).norm(), 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(NumericalEquivalenceWithOldMobileFilter)
{
    const MobileReachy2 robot;
    const auto q = midConfiguration<MobileReachy2>();
    const Barrier<MobileReachy2> barrier(robot, nearField(), q);
    const auto lower = MobileReachy2::lowerBounds();
    const auto upper = MobileReachy2::upperBounds();
    const Filter<MobileReachy2> filter(barrier, lower, upper, 0.001);

    for (const auto &nominal :
         {MobileReachy2::Configuration::Zero().eval(), towardWorstSphere(barrier, q, 2.0),
          towardWorstSphere(barrier, q, 10.0), MobileReachy2::Configuration::Constant(50.0).eval(),
          MobileReachy2::Configuration::Constant(-1.5).eval()})
    {
        MobileReachy2::Configuration filtered;
        filter.filter(q, nominal, duration, filtered);
        const MobileReachy2::Configuration expected = oracle::mobileFilter(barrier, q, nominal, duration, lower, upper);
        BOOST_CHECK_LE((filtered - expected).norm(), 1e-9);
    }
}
