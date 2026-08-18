// Coupled 14-DoF Reachy2 planning: collision-checked RRTConnect versus
// CBF-rollout RRTConnect with certified coarse steps.
//
// The robot geometry/kinematics in ompl/robots/Reachy2.h is generated from the
// supplied spherized URDF and SRDF by scripts/generate_reachy2_model.py.
//
// Usage:
//   demo_Reachy2CBFPlanning [seconds] [out.path]
//       [left_x left_y left_z right_x right_y right_z] [shortcut_radians] [trials]
//   demo_Reachy2CBFPlanning [seconds] [out.path] [shortcut_radians] [trials]

// The output is one raw-radian configuration per row, in Reachy2::jointNames()
// order. By default both hands move into the shelf's lower bay.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/base/PlannerData.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/ControlFilter.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/RobotCBFControlFilter.h>
#include <ompl/cbf/RobotClearanceBarrier.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/robots/Reachy2.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using Robot = ompl::robots::Reachy2;
using Configuration = Robot::Configuration;
using Barrier = ompl::cbf::RobotClearanceBarrier<Robot>;
using Filter = ompl::cbf::RobotCBFControlFilter<Robot>;
constexpr int N = static_cast<int>(Robot::nJoints);

namespace
{
    struct Box
    {
        Eigen::Vector3d center;
        Eigen::Vector3d half;
    };

    double boxDistance(const Eigen::Vector3d &p, const Box &b)
    {
        const Eigen::Vector3d q = (p - b.center).cwiseAbs() - b.half;
        return q.cwiseMax(0.0).norm() + std::min(q.maxCoeff(), 0.0);
    }

    std::vector<Box> shelf()
    {
        // Reuse the UR5 shelf's useful bay coordinates, but not its 0.9144 m
        // robot-mounting table. Reachy is a floor-standing mobile manipulator;
        // enclosing it in a hole in the UR5 table makes the robot look buried
        // and would prevent the base from moving in a later mobile RRT state.
        // Extend the shelf's back and sides to the floor so it is free-standing.
        constexpr double bottomShelf = 0.9144;
        constexpr double shelfX = 0.62, depth = 0.14, width = 0.75;
        constexpr double bayPitch = 0.44, originalPanelHalf = 0.46;
        constexpr double top = bottomShelf + 2 * originalPanelHalf;
        std::vector<Box> boxes{
            {{shelfX + depth, 0, top / 2}, {0.02, width / 2, top / 2}},
            {{shelfX, width / 2, top / 2}, {depth, 0.02, top / 2}},
            {{shelfX, -width / 2, top / 2}, {depth, 0.02, top / 2}},
        };
        for (double z : {0.0, bayPitch, 2 * bayPitch})
            boxes.push_back({{shelfX, 0, bottomShelf + z}, {depth, width / 2, 0.015}});
        return boxes;
    }

    class SeededRealVectorSampler : public ob::RealVectorStateSampler
    {
    public:
        SeededRealVectorSampler(const ob::StateSpace *space, std::uint_fast32_t seed)
          : ob::RealVectorStateSampler(space)
        {
            rng_.setLocalSeed(seed);
        }
    };

    class SeededRRTConnect : public og::RRTConnect
    {
    public:
        SeededRRTConnect(const ob::SpaceInformationPtr &si, std::uint_fast32_t seed)
          : og::RRTConnect(si)
        {
            rng_.setLocalSeed(seed);
        }
    };

    bool solveIK(const Robot &robot, const Barrier &barrier, const Eigen::Vector3d &left,
                 const Eigen::Vector3d &right, Configuration &solution,
                 const Configuration *initialGuess = nullptr)
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const Configuration lo = Robot::lowerBounds(), hi = Robot::upperBounds();
        double best = std::numeric_limits<double>::infinity();
        double bestAny = std::numeric_limits<double>::infinity();
        Configuration bestQ = Configuration::Zero();
        for (int restart = 0; restart < 80; ++restart)
        {
            Configuration q;
            if (restart == 0)
                q = initialGuess ? *initialGuess : 0.5 * (lo + hi);
            else
                for (int j = 0; j < N; ++j)
                    q[j] = lo[j] + unit(rng) * (hi[j] - lo[j]);

            for (int iter = 0; iter < 180; ++iter)
            {
                const auto kin = robot.kinematics(q);
                Eigen::Matrix<double, 6, 1> error;
                error.head<3>() = left - Robot::tipPosition(kin, true);
                error.tail<3>() = right - Robot::tipPosition(kin, false);
                bestAny = std::min(bestAny, error.norm());
                if (error.norm() < 0.01 && barrier.safe(q))
                {
                    solution = q;
                    return true;
                }
                Eigen::Matrix<double, 6, N> J;
                J.topRows<3>() = Robot::tipJacobian(kin, true);
                J.bottomRows<3>() = Robot::tipJacobian(kin, false);
                constexpr double damping = 2e-3;
                const Eigen::Matrix<double, N, 1> dq =
                    J.transpose() * (J * J.transpose() + damping * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(error);
                q += 0.45 * dq;
                q = q.cwiseMax(lo).cwiseMin(hi);
                if (error.norm() < best && barrier.safe(q))
                {
                    best = error.norm();
                    bestQ = q;
                }
            }
        }
        solution = bestQ;
        std::fprintf(stderr, "IK diagnostics: best Cartesian error %.4f m, best safe error %.4f m\n",
                     bestAny, best);
        return best < 0.05;
    }

    Configuration stateConfiguration(const ob::State *state)
    {
        const double *v = state->as<ob::RealVectorStateSpace::StateType>()->values;
        Configuration q;
        for (int j = 0; j < N; ++j)
            q[j] = v[j];
        return q;
    }

    constexpr double integrationStep = 0.02;

    double arcLength(const std::vector<Configuration> &motion)
    {
        double length = 0.0;
        for (std::size_t i = 1; i < motion.size(); ++i)
            length += (motion[i] - motion[i - 1]).norm();
        return length;
    }

    double median(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        return values.size() % 2 ? values[middle]
                                 : 0.5 * (values[middle - 1] + values[middle]);
    }

    struct PlanResult
    {
        bool solved{false};
        bool exact{false};
        bool safe{false};
        double seconds{0.0};
        std::size_t evaluations{0};
        std::size_t vertices{0};
        std::size_t unsafeStates{0};
        std::size_t qpCalls{0};
        std::size_t coarseSteps{0};
        std::size_t misses{0};
        double meanActiveRows{0.0};
        double travel{0.0};
        double worstWorld{std::numeric_limits<double>::infinity()};
        double worstSelf{std::numeric_limits<double>::infinity()};
        double leftError{std::numeric_limits<double>::infinity()};
        double rightError{std::numeric_limits<double>::infinity()};
        std::vector<Configuration> motion;
    };

    void auditMotion(PlanResult &result, const Robot &robot, const Barrier &barrier,
                     const Eigen::Vector3d &left, const Eigen::Vector3d &right)
    {
        if (result.motion.empty())
            return;
        result.safe = true;
        result.unsafeStates = 0;
        result.worstWorld = std::numeric_limits<double>::infinity();
        result.worstSelf = std::numeric_limits<double>::infinity();
        for (const Configuration &q : result.motion)
        {
            double world = 0.0, self = 0.0;
            barrier.safe(q, &world, &self);
            result.worstWorld = std::min(result.worstWorld, world);
            result.worstSelf = std::min(result.worstSelf, self);
            if (world < -0.001 || self < -0.001)
            {
                result.safe = false;
                ++result.unsafeStates;
            }
        }
        const auto endKin = robot.kinematics(result.motion.back());
        result.leftError = (Robot::tipPosition(endKin, true) - left).norm();
        result.rightError = (Robot::tipPosition(endKin, false) - right).norm();
        result.solved = result.safe && result.leftError <= 0.04 && result.rightError <= 0.04;
    }

    void setBounds(const std::shared_ptr<ob::RealVectorStateSpace> &space);

    std::vector<Configuration> shortcutBaseline(
        const std::vector<Configuration> &motion, const Barrier &barrier,
        double delta, double equivalenceTolerance, double auditSpacing,
        double &lengthBefore, double &lengthAfter, double &seconds)
    {
        auto space = std::make_shared<ob::RealVectorStateSpace>(N);
        setBounds(space);
        space->setLongestValidSegmentFraction(auditSpacing / space->getMaximumExtent());
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(
            [&barrier](const ob::State *state)
            { return barrier.safe(stateConfiguration(state)); });
        si->setup();

        og::PathGeometric path(si);
        ob::ScopedState<ob::RealVectorStateSpace> state(space);
        for (const Configuration &q : motion)
        {
            for (int j = 0; j < N; ++j)
                state[j] = q[j];
            path.append(state.get());
        }

        lengthBefore = path.length();
        const auto begin = std::chrono::steady_clock::now();
        og::PathSimplifier(si).ropeShortcutPath(path, delta, equivalenceTolerance);
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        lengthAfter = path.length();

        const unsigned int states = std::max(
            2u, static_cast<unsigned int>(std::ceil(lengthAfter / auditSpacing)) + 1u);
        path.interpolate(states);
        std::vector<Configuration> result;
        result.reserve(path.getStateCount());
        for (std::size_t i = 0; i < path.getStateCount(); ++i)
            result.push_back(stateConfiguration(path.getState(i)));
        return result;
    }

    void setBounds(const std::shared_ptr<ob::RealVectorStateSpace> &space)
    {
        ob::RealVectorBounds bounds(N);
        const Configuration lo = Robot::lowerBounds();
        const Configuration hi = Robot::upperBounds();
        for (int j = 0; j < N; ++j)
        {
            bounds.setLow(j, lo[j]);
            bounds.setHigh(j, hi[j]);
        }
        space->setBounds(bounds);
    }

    PlanResult runCBF(const Robot &robot, const Barrier &barrier,
                      const Configuration &start, const Configuration &goal,
                      const Eigen::Vector3d &left, const Eigen::Vector3d &right,
                      double timeLimit, std::uint_fast32_t seed, double auditSpacing)
    {
        Filter filter(barrier);
        using Space = ompl::cbf::RobotFilteredStateSpace<Robot>;
        auto space = std::make_shared<Space>(filter, integrationStep, Filter::maxSpeed());
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *stateSpace)
            { return std::make_shared<SeededRealVectorSampler>(stateSpace, seed + 1); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(
            std::make_shared<ompl::cbf::RobotFilteredMotionValidator<Robot>>(si));
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> startState(space), goalState(space);
        for (int j = 0; j < N; ++j)
        {
            startState[j] = start[j];
            goalState[j] = goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(startState, goalState, 0.05);
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(1.5);
        planner->setProblemDefinition(pdef);
        planner->setup();

        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status =
            planner->solve(ob::timedPlannerTerminationCondition(timeLimit));

        PlanResult result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.exact = status == ob::PlannerStatus::EXACT_SOLUTION;
        const auto stats = space->statistics();
        result.evaluations = stats.steps;
        result.coarseSteps = stats.coarse;
        result.travel = stats.travel;
        result.qpCalls = filter.qpCalls();
        result.meanActiveRows = filter.meanActiveRows();
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();
        if (!status || !pdef->hasSolution())
            return result;

        const auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        const og::PathGeometric executed =
            ompl::cbf::robotExecutedPath<Robot>(*solution, auditSpacing, &result.misses);
        result.motion.reserve(executed.getStateCount());
        for (std::size_t i = 0; i < executed.getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(executed.getState(i)));
        auditMotion(result, robot, barrier, left, right);
        return result;
    }

    PlanResult runBaseline(const Robot &robot, const Barrier &barrier,
                           const Configuration &start, const Configuration &goal,
                           const Eigen::Vector3d &left, const Eigen::Vector3d &right,
                           double timeLimit, std::uint_fast32_t seed, double auditSpacing)
    {
        auto space = std::make_shared<ob::RealVectorStateSpace>(N);
        setBounds(space);
        space->setLongestValidSegmentFraction(auditSpacing / space->getMaximumExtent());
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *stateSpace)
            { return std::make_shared<SeededRealVectorSampler>(stateSpace, seed + 1); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        std::size_t checks = 0;
        si->setStateValidityChecker(
            [&barrier, &checks](const ob::State *state)
            {
                ++checks;
                return barrier.safe(stateConfiguration(state));
            });
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> startState(space), goalState(space);
        for (int j = 0; j < N; ++j)
        {
            startState[j] = start[j];
            goalState[j] = goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(startState, goalState, 0.05);
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(1.5);
        planner->setProblemDefinition(pdef);
        planner->setup();

        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status =
            planner->solve(ob::timedPlannerTerminationCondition(timeLimit));

        PlanResult result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.exact = status == ob::PlannerStatus::EXACT_SOLUTION;
        result.evaluations = checks;
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();
        if (!status || !pdef->hasSolution())
            return result;

        auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        const unsigned int states = std::max(
            2u, static_cast<unsigned int>(std::ceil(solution->length() / auditSpacing)) + 1u);
        solution->interpolate(states);
        result.motion.reserve(solution->getStateCount());
        for (std::size_t i = 0; i < solution->getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(solution->getState(i)));
        auditMotion(result, robot, barrier, left, right);
        return result;
    }

    void writePath(const std::string &path, const std::vector<Configuration> &motion)
    {
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("cannot open output path " + path);
        out.precision(17);
        out << "# joints";
        for (const char *name : Robot::jointNames())
            out << ' ' << name;
        out << '\n';
        for (const Configuration &q : motion)
        {
            for (int j = 0; j < N; ++j)
                out << (j ? " " : "") << q[j];
            out << '\n';
        }
    }

    std::string baselinePathFor(const std::string &cbfPath)
    {
        const std::string suffix = "_cbf.path";
        if (cbfPath.size() >= suffix.size() &&
            cbfPath.compare(cbfPath.size() - suffix.size(), suffix.size(), suffix) == 0)
            return cbfPath.substr(0, cbfPath.size() - suffix.size()) + "_ompl.path";
        return cbfPath + ".ompl";
    }

}  // namespace

int main(int argc, char **argv)
{
    ompl::RNG::setSeed(7);
    const double seconds = argc > 1 ? std::stod(argv[1]) : 10.0;
    const std::string output = argc > 2 ? argv[2] : "reachy2_cbf.path";
    // Fine anchors and a low savings threshold make the default shortcut pass
    // deliberately aggressive without relaxing its rollout or arrival checks.
    double shortcutDelta = 0.05;
    int trials = 5;
    constexpr double shortcutEquivalenceTolerance = 0.01;
    // Default: both hands enter the lower bay. Both targets can be replaced by
    // the six CLI coordinates.
    Eigen::Vector3d left(0.62, 0.20, 1.1344), right(0.62, -0.20, 1.1344);
    if (argc >= 9 && argc <= 11)
    {
        for (int i = 0; i < 3; ++i)
            left[i] = std::stod(argv[3 + i]);
        for (int i = 0; i < 3; ++i)
            right[i] = std::stod(argv[6 + i]);
        if (argc >= 10)
            shortcutDelta = std::stod(argv[9]);
        if (argc == 11)
            trials = std::stoi(argv[10]);
    }
    else if (argc == 4 || argc == 5)
    {
        shortcutDelta = std::stod(argv[3]);
        if (argc == 5)
            trials = std::stoi(argv[4]);
    }
    else if (argc != 1 && argc != 2 && argc != 3)
    {
        std::fprintf(stderr,
                     "usage: %s [seconds] [out.path] [lx ly lz rx ry rz] "
                     "[shortcutRadians] [trials]\n"
                     "       %s [seconds] [out.path] [shortcutRadians] [trials]\n",
                     argv[0],
                     argv[0]);
        return 2;
    }
    if (shortcutDelta < 0.0)
    {
        std::fprintf(stderr, "shortcutRadians must be non-negative (zero disables it)\n");
        return 2;
    }
    if (trials < 1)
    {
        std::fprintf(stderr, "trials must be at least one\n");
        return 2;
    }

    const std::vector<Box> boxes = shelf();
    const auto sceneDistance = [&boxes](const Eigen::Vector3d &p)
    {
        double d = std::numeric_limits<double>::infinity();
        for (const Box &b : boxes)
            d = std::min(d, boxDistance(p, b));
        return d;
    };
    // Same obstacles, enlarged bake bounds for Reachy's wider static body.
    const Eigen::AlignedBox3d workspace(Eigen::Vector3d(-1.5, -1.5, -0.5),
                                        Eigen::Vector3d(1.5, 1.5, 2.40));
    std::printf("Baking shelf SDF...\n");
    const ompl::sdf::GridSDF field(sceneDistance, workspace, 0.025);
    const Robot robot;
    const Configuration reference = 0.5 * (Robot::lowerBounds() + Robot::upperBounds());
    const Barrier startSearchBarrier(robot, field, reference);
    Configuration start;
    if (!solveIK(robot, startSearchBarrier, Eigen::Vector3d(0.25, 0.20, 1.1344),
                 Eigen::Vector3d(0.25, -0.20, 1.1344), start))
    {
        std::fprintf(stderr, "could not construct the safe pre-shelf start pose\n");
        return 3;
    }
    const Barrier barrier(robot, field, start);
    const auto startKin = robot.kinematics(start);
    const Eigen::Vector3d startLeft = Robot::tipPosition(startKin, true);
    const Eigen::Vector3d startRight = Robot::tipPosition(startKin, false);
    std::printf("start-pose tips: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n",
                startLeft.x(), startLeft.y(), startLeft.z(), startRight.x(), startRight.y(), startRight.z());
    double startWorld = 0.0, startSelf = 0.0;
    if (!barrier.safe(start, &startWorld, &startSelf))
    {
        std::fprintf(stderr, "declared start is unsafe: world=%+.4f self=%+.4f\n", startWorld, startSelf);
        return 3;
    }
    Configuration goal;
    if (!solveIK(robot, barrier, left, right, goal, &start))
    {
        std::fprintf(stderr, "no safe bimanual IK solution for the requested hand targets\n");
        return 4;
    }

    const auto goalKin = robot.kinematics(goal);
    std::printf("Reachy2: %d joints, %zu spheres, %zu/%zu semantic self-pairs enabled\n",
                N, Robot::nSpheres, barrier.enabledSelfPairs(), Robot::nSelfPairs);
    std::printf("left target/reached:  %.3f %.3f %.3f / %.3f %.3f %.3f\n",
                left.x(), left.y(), left.z(), Robot::tipPosition(goalKin, true).x(),
                Robot::tipPosition(goalKin, true).y(), Robot::tipPosition(goalKin, true).z());
    std::printf("right target/reached: %.3f %.3f %.3f / %.3f %.3f %.3f\n",
                right.x(), right.y(), right.z(), Robot::tipPosition(goalKin, false).x(),
                Robot::tipPosition(goalKin, false).y(), Robot::tipPosition(goalKin, false).z());

    const Configuration speed = Robot::velocityLimits().cwiseMin(Configuration::Constant(1.2));
    const double auditSpacing = integrationStep * speed.maxCoeff();
    std::vector<double> cbfTimes, omplTimes, cbfEvaluations, omplEvaluations,
        cbfVertices, omplVertices;
    std::size_t cbfSolved = 0, omplSolved = 0;
    PlanResult cbfResult, omplResult;
    bool haveCbfPath = false, haveOmplPath = false;

    std::printf("\nPer-trial planner results (nodes are PlannerData tree vertices)\n");
    std::printf("%5s %12s %12s %9s %12s %12s %9s\n",
                "trial", "OMPL ms", "OMPL nodes", "OMPL ok", "CBF ms", "CBF nodes", "CBF ok");

    // Alternate the order so cache and frequency-scaling effects are not always
    // awarded to the same row. SDF construction and IK are deliberately outside
    // this loop and outside the timed region.
    for (int trial = 0; trial < trials; ++trial)
    {
        const std::uint_fast32_t trialSeed = 7001u + 17u * static_cast<std::uint_fast32_t>(trial);
        PlanResult cbfRun, omplRun;
        if (trial % 2 == 0)
        {
            cbfRun = runCBF(robot, barrier, start, goal, left, right, seconds,
                            trialSeed, auditSpacing);
            omplRun = runBaseline(robot, barrier, start, goal, left, right, seconds,
                                  trialSeed, auditSpacing);
        }
        else
        {
            omplRun = runBaseline(robot, barrier, start, goal, left, right, seconds,
                                  trialSeed, auditSpacing);
            cbfRun = runCBF(robot, barrier, start, goal, left, right, seconds,
                            trialSeed, auditSpacing);
        }

        std::printf("%5d %12.3f %12zu %9s %12.3f %12zu %9s\n",
                    trial + 1, 1e3 * omplRun.seconds, omplRun.vertices,
                    omplRun.solved ? "yes" : "no", 1e3 * cbfRun.seconds,
                    cbfRun.vertices, cbfRun.solved ? "yes" : "no");

        cbfTimes.push_back(cbfRun.seconds);
        omplTimes.push_back(omplRun.seconds);
        cbfEvaluations.push_back(static_cast<double>(cbfRun.evaluations));
        omplEvaluations.push_back(static_cast<double>(omplRun.evaluations));
        cbfVertices.push_back(static_cast<double>(cbfRun.vertices));
        omplVertices.push_back(static_cast<double>(omplRun.vertices));
        cbfSolved += cbfRun.solved ? 1 : 0;
        omplSolved += omplRun.solved ? 1 : 0;
        if (cbfRun.solved || !haveCbfPath)
        {
            haveCbfPath = cbfRun.solved;
            cbfResult = std::move(cbfRun);
        }
        if (omplRun.solved || !haveOmplPath)
        {
            haveOmplPath = omplRun.solved;
            omplResult = std::move(omplRun);
        }
    }

    const double cbfMedian = median(cbfTimes);
    const double omplMedian = median(omplTimes);
    std::printf("\nPlanner wall time only (SDF bake, IK, audit, output, and shortcut excluded)\n");
    std::printf("paired seeds; geometric RRTConnect range 1.5 rad; matched %.4f rad "
                "baseline checks / path audit\n",
                auditSpacing);
    std::printf("%-12s %9s %12s %12s %12s %10s %10s %10s\n",
                "row", "solved", "median ms", "median evals", "median nodes", "path rad",
                "world m", "self m");
    std::printf("%-12s %4zu/%-4d %12.3f %12.0f %12.0f %10.3f %+10.4f %+10.4f\n",
                "ompl-rrtc", omplSolved, trials, 1e3 * omplMedian, median(omplEvaluations),
                median(omplVertices), haveOmplPath ? arcLength(omplResult.motion) : 0.0,
                haveOmplPath ? omplResult.worstWorld : 0.0,
                haveOmplPath ? omplResult.worstSelf : 0.0);
    std::printf("%-12s %4zu/%-4d %12.3f %12.0f %12.0f %10.3f %+10.4f %+10.4f\n",
                "cbf-rrtc", cbfSolved, trials, 1e3 * cbfMedian, median(cbfEvaluations),
                median(cbfVertices), haveCbfPath ? arcLength(cbfResult.motion) : 0.0,
                haveCbfPath ? cbfResult.worstWorld : 0.0,
                haveCbfPath ? cbfResult.worstSelf : 0.0);
    if (omplMedian <= cbfMedian)
        std::printf("wall time: OMPL is %.2fx faster (ompl/cbf = %.2f)\n",
                    omplMedian > 0.0 ? cbfMedian / omplMedian : 0.0,
                    cbfMedian > 0.0 ? omplMedian / cbfMedian : 0.0);
    else
        std::printf("wall time: CBF is %.2fx faster (ompl/cbf = %.2f)\n",
                    cbfMedian > 0.0 ? omplMedian / cbfMedian : 0.0,
                    cbfMedian > 0.0 ? omplMedian / cbfMedian : 0.0);
    std::printf("evals are collision checks for ompl-rrtc and filter calls for cbf-rrtc\n");
    if (haveCbfPath)
        std::printf("cbf representative: %zu/%zu calls entered QP, %.2f active rows/call, "
                    "%.1f%% certified coarse steps, %.4f rad/call, %zu replay misses\n",
                    cbfResult.qpCalls, cbfResult.evaluations, cbfResult.meanActiveRows,
                    cbfResult.evaluations > 0
                        ? 1e2 * cbfResult.coarseSteps / cbfResult.evaluations
                        : 0.0,
                    cbfResult.evaluations > 0
                        ? cbfResult.travel / cbfResult.evaluations
                        : 0.0,
                    cbfResult.misses);
    std::printf("path/clearance columns describe the final successful trial for each row\n");

    if (!haveCbfPath)
    {
        std::fprintf(stderr, "no audited CBF-safe plan found in %d trial(s)\n", trials);
        return 5;
    }

    std::vector<Configuration> cbfMotion = cbfResult.motion;
    std::vector<Configuration> omplMotion = omplResult.motion;
    ompl::cbf::ShortcutReport cbfShortcut;
    double cbfShortcutSeconds = 0.0;
    if (shortcutDelta > 0.0)
    {
        if (haveOmplPath)
        {
            double before = 0.0, after = 0.0, shortcutSeconds = 0.0;
            omplMotion = shortcutBaseline(omplMotion, barrier, shortcutDelta,
                                          shortcutEquivalenceTolerance, auditSpacing,
                                          before, after, shortcutSeconds);
            omplResult.motion = omplMotion;
            auditMotion(omplResult, robot, barrier, left, right);
            std::printf("OMPL shortcut: %.4f -> %.4f rad, %.1f ms\n",
                        before, after, 1e3 * shortcutSeconds);
            if (!omplResult.solved)
            {
                std::fprintf(stderr, "OMPL shortcut failed its dense safety/goal audit\n");
                return 6;
            }
        }

        Filter shortcutFilter(barrier);
        ompl::cbf::RobotFilteredStateSpace<Robot> shortcutSpace(
            shortcutFilter, integrationStep, speed);
        const auto begin = std::chrono::steady_clock::now();
        const double arrivalTolerance = 0.05 * speed.norm() * integrationStep;
        const auto rollout = [&shortcutSpace](const Configuration &from,
                                              const Configuration &to)
        {
            auto result = shortcutSpace.roll(from, to, 1.0);
            return result.fraction >= 1.0 - 1e-12
                       ? std::move(result.waypoints)
                       : std::vector<Configuration>();
        };
        cbfMotion = ompl::cbf::ropeShortcut(cbfMotion, shortcutDelta, arrivalTolerance,
                                            rollout, &cbfShortcut, shortcutEquivalenceTolerance);
        cbfShortcutSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("CBF shortcut: %.4f -> %.4f rad, %zu accepted from %zu rollouts, "
                    "%zu screened, %.1f ms, max arrival gap %.3f mm\n",
                    cbfShortcut.lengthBefore, cbfShortcut.lengthAfter, cbfShortcut.accepted,
                    cbfShortcut.rollouts, cbfShortcut.screened, 1e3 * cbfShortcutSeconds,
                    1e3 * cbfShortcut.maxArrivalGap);
    }
    double worstWorld = std::numeric_limits<double>::infinity();
    double worstSelf = std::numeric_limits<double>::infinity();
    for (const Configuration &q : cbfMotion)
    {
        double w = 0.0, s = 0.0;
        barrier.safe(q, &w, &s);
        // The QP linearizes a finite step. The 10 mm modeled margin is the
        // physical safety claim; allow 1 mm of that guard to absorb the same
        // endpoint overshoot documented for the UR5 CBF implementation.
        if (w < -0.001 || s < -0.001)
        {
            std::fprintf(stderr, "internal audit failed: world=%+.6f self=%+.6f\n", w, s);
            return 6;
        }
        worstWorld = std::min(worstWorld, w);
        worstSelf = std::min(worstSelf, s);
    }
    writePath(output, cbfMotion);
    const std::string omplOutput = baselinePathFor(output);
    if (haveOmplPath)
        writePath(omplOutput, omplMotion);
    std::printf("Solved: %zu audited states, worst world=%+.4f m, self=%+.4f m\n",
                cbfMotion.size(), worstWorld, worstSelf);
    const auto endKin = robot.kinematics(cbfMotion.back());
    const Eigen::Vector3d endLeft = Robot::tipPosition(endKin, true);
    const Eigen::Vector3d endRight = Robot::tipPosition(endKin, false);
    const double leftError = (endLeft - left).norm();
    const double rightError = (endRight - right).norm();
    std::printf("Final tips: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n",
                endLeft.x(), endLeft.y(), endLeft.z(), endRight.x(), endRight.y(), endRight.z());
    std::printf("Cartesian errors: left %.4f m, right %.4f m (%s joint-space solution)\n",
                leftError, rightError, cbfResult.exact ? "exact" : "approximate");
    // The requested goal is Cartesian; an approximate connection to one IK
    // representative is still a valid answer when both actual tips reach it.
    if (leftError > 0.04 || rightError > 0.04)
    {
        std::fprintf(stderr, "solution did not reach both Cartesian goals within 4 cm\n");
        return 7;
    }
    std::printf("CBF path: %s\n", output.c_str());
    if (haveOmplPath)
        std::printf("OMPL path: %s\n", omplOutput.c_str());
    return 0;
}
