#define BOOST_TEST_MODULE Reachy2MobileTest
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/cbf/ControlFilter.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/robots/HolonomicMobileManipulator.h>
#include <ompl/robots/Reachy2.h>

using Arm = ompl::robots::Reachy2;
using Mobile = ompl::robots::HolonomicMobileManipulator<Arm>;
using Configuration = Mobile::Configuration;
using Operations = Mobile::ConfigurationOperations;

namespace
{
    Configuration configuration()
    {
        Configuration q = Configuration::Zero();
        q.template tail<Arm::nJoints>() =
            0.5 * (Arm::lowerBounds() + Arm::upperBounds());
        return q;
    }
}

BOOST_AUTO_TEST_CASE(IdentityBaseMatchesFixedReachy)
{
    const Arm fixed;
    const Mobile mobile(fixed);
    const Configuration q = configuration();
    const auto fixedKin = fixed.kinematics(Mobile::armConfiguration(q));
    const auto mobileKin = mobile.kinematics(q);

    for (std::size_t i = 0; i < Arm::nSpheres; ++i)
    {
        BOOST_CHECK_SMALL((Arm::sphereCenter(fixedKin, i) -
                           Mobile::sphereCenter(mobileKin, i)).norm(), 1e-12);
        BOOST_CHECK_SMALL((Arm::sphereJacobian(fixedKin, i) -
                           Mobile::sphereJacobian(mobileKin, i)
                               .template rightCols<Arm::nJoints>()).norm(), 1e-12);
    }
    BOOST_CHECK_SMALL((Arm::tipPosition(fixedKin, true) -
                       Mobile::tipPosition(mobileKin, true)).norm(), 1e-12);
    BOOST_CHECK_SMALL((Arm::tipPosition(fixedKin, false) -
                       Mobile::tipPosition(mobileKin, false)).norm(), 1e-12);
}

BOOST_AUTO_TEST_CASE(SphereAndTipJacobiansMatchFiniteDifferences)
{
    const Mobile robot;
    Configuration q = configuration();
    q.template head<3>() << -0.23, 0.17, 0.41;
    const auto kin = robot.kinematics(q);
    constexpr double step = 1e-6;

    for (const std::size_t sphere : {std::size_t{0}, std::size_t{25}, std::size_t{70}})
    {
        const Mobile::Jacobian analytic = Mobile::sphereJacobian(kin, sphere);
        for (const int column : {0, 1, 2, 3, 8, 10, 16})
        {
            Configuration offset = Configuration::Zero();
            offset[column] = step;
            const Eigen::Vector3d numeric =
                (Mobile::sphereCenter(robot.kinematics(q + offset), sphere) -
                 Mobile::sphereCenter(robot.kinematics(q - offset), sphere)) /
                (2.0 * step);
            BOOST_CHECK_SMALL((analytic.col(column) - numeric).norm(), 2e-7);
        }
    }

    for (const bool left : {false, true})
    {
        const Mobile::Jacobian analytic = Mobile::tipJacobian(kin, left);
        for (const int column : {0, 1, 2, 3, 8, 10, 16})
        {
            Configuration offset = Configuration::Zero();
            offset[column] = step;
            const Eigen::Vector3d numeric =
                (Mobile::tipPosition(robot.kinematics(q + offset), left) -
                 Mobile::tipPosition(robot.kinematics(q - offset), left)) /
                (2.0 * step);
            BOOST_CHECK_SMALL((analytic.col(column) - numeric).norm(), 2e-7);
        }
    }
}

BOOST_AUTO_TEST_CASE(SelfDistancesAndDerivativesIgnoreRigidBaseMotion)
{
    const Mobile robot;
    Configuration a = configuration();
    Configuration b = a;
    b.template head<3>() << 0.31, -0.22, 1.13;
    const auto ka = robot.kinematics(a);
    const auto kb = robot.kinematics(b);

    for (std::size_t p = 0; p < Mobile::nSelfPairs; p += 31)
    {
        const auto pair = Mobile::selfPairs()[p];
        const Eigen::Vector3d da = Mobile::sphereCenter(ka, pair.a) -
                                   Mobile::sphereCenter(ka, pair.b);
        const Eigen::Vector3d db = Mobile::sphereCenter(kb, pair.a) -
                                   Mobile::sphereCenter(kb, pair.b);
        BOOST_CHECK_SMALL(std::abs(da.norm() - db.norm()), 1e-12);

        const auto ja = Mobile::sphereJacobian(ka, pair.a);
        const auto jb = Mobile::sphereJacobian(ka, pair.b);
        if (da.norm() > 1e-12)
            BOOST_CHECK_SMALL(((da / da.norm()).transpose() * (ja - jb))
                                  .template head<3>().norm(), 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(PeriodicYawOperationsTakeTheShortArc)
{
    Configuration a = Configuration::Zero(), b = Configuration::Zero();
    constexpr double pi = 3.14159265358979323846;
    a[2] = pi - 0.05;
    b[2] = -pi + 0.05;
    const Configuration speed = Mobile::velocityLimits();

    BOOST_CHECK_CLOSE(Operations::difference(a, b)[2], 0.1, 1e-10);
    BOOST_CHECK_CLOSE(Operations::duration(a, b, speed), 0.1 / speed[2], 1e-10);
    const Configuration middle = Operations::interpolate(a, b, 0.5);
    BOOST_CHECK_SMALL(std::abs(std::abs(middle[2]) - pi), 1e-12);
    BOOST_CHECK_SMALL(Operations::difference(Operations::integrate(a,
        Operations::difference(a, b), 1.0), b).norm(), 1e-12);
}

BOOST_AUTO_TEST_CASE(WrappedYawSurvivesRolloutLedgerReplayAndShortcutDensification)
{
    ompl::cbf::RobotPassthroughFilter<Mobile> filter;
    using Space = ompl::cbf::RobotFilteredStateSpace<Mobile>;
    auto space = std::make_shared<Space>(filter, 0.02, Mobile::velocityLimits());
    auto si = std::make_shared<ompl::base::SpaceInformation>(space);
    si->setup();

    Configuration a = configuration(), b = a, c = a;
    constexpr double pi = 3.14159265358979323846;
    a[2] = pi - 0.08;
    b[2] = -pi + 0.02;
    c[2] = -pi + 0.12;

    const Space::Rollout rolled = space->roll(a, b, 1.0);
    BOOST_REQUIRE(rolled.reachedTarget);
    BOOST_CHECK_SMALL(Operations::difference(rolled.end, b).norm(), 1e-12);
    space->record(a, b, rolled.waypoints);

    ompl::geometric::PathGeometric path(si);
    ompl::base::ScopedState<> state(space);
    Space::setState(state.get(), a);
    path.append(state.get());
    Space::setState(state.get(), b);
    path.append(state.get());
    std::size_t misses = 99;
    const auto replay = ompl::cbf::robotExecutedPath<Mobile>(path, 0.01, &misses);
    BOOST_CHECK_EQUAL(misses, 0u);
    BOOST_CHECK_GT(replay.getStateCount(), 2u);
    for (std::size_t i = 1; i < replay.getStateCount(); ++i)
    {
        const Configuration previous = Space::configurationOf(replay.getState(i - 1));
        const Configuration current = Space::configurationOf(replay.getState(i));
        BOOST_CHECK_LT(std::abs(Operations::difference(previous, current)[2]), 0.02);
    }

    std::vector<Configuration> motion{a, b, c};
    const auto rollout = [](const Configuration &from, const Configuration &to)
    {
        return std::vector<Configuration>{from, Operations::interpolate(from, to, 0.5), to};
    };
    const auto shortened = ompl::cbf::ropeShortcut(
        motion, 0.05, 0.001, rollout, nullptr, 0.0, Operations());
    BOOST_REQUIRE_GE(shortened.size(), 2u);
    for (std::size_t i = 1; i < shortened.size(); ++i)
        BOOST_CHECK_LT(std::abs(Operations::difference(shortened[i - 1], shortened[i])[2]), 0.2);
}
