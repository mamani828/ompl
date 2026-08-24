// A focused visual comparison of straight-edge RRTConnect and one CBF rollout
// for the spherized UR5.  The companion PyBullet viewer is
// scripts/visualize_ur5_cbf_curve.py.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/robots/UR5.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using Barrier = ompl::cbf::ClearanceBarrier;
using Filter = ompl::cbf::CBFControlFilter;
using UR5 = ompl::robots::UR5;

namespace
{
    constexpr int dimension = static_cast<int>(UR5::nJoints);
    constexpr double stepSize = 0.05;

    UR5::Configuration startConfiguration()
    {
        return (UR5::Configuration() << 0.0, -1.2, 1.8, -0.6, 1.57, 0.0).finished();
    }

    UR5::Configuration goalConfiguration()
    {
        return (UR5::Configuration() << 2.4, -1.2, 1.8, -0.6, 1.57, 0.0).finished();
    }

    const std::array<Eigen::Vector3d, 2> obstacleCenters{
        Eigen::Vector3d(-0.742, 0.121, 1.083),
        Eigen::Vector3d(-0.742, 0.121, 1.383)};
    constexpr std::array<double, 2> obstacleRadii{0.18, 0.14};

    ompl::sdf::DistanceFn scene()
    {
        return [](const Eigen::Vector3d &point)
        {
            double distance = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < obstacleCenters.size(); ++i)
                distance = std::min(distance, (point - obstacleCenters[i]).norm() - obstacleRadii[i]);
            return distance;
        };
    }

    UR5::Configuration configurationOf(const ob::State *state)
    {
        const double *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
        UR5::Configuration q;
        for (int j = 0; j < dimension; ++j)
            q[j] = values[j];
        return q;
    }

    std::vector<UR5::Configuration> baselinePath(const ompl::sdf::GridSDF &field,
                                                  std::vector<UR5::Configuration> &nodes)
    {
        const UR5 robot;
        const Barrier barrier(robot, field, Barrier::defaultMargin);
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        ob::RealVectorBounds bounds(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, UR5::lowerBounds()[j]);
            bounds.setHigh(j, UR5::upperBounds()[j]);
        }
        space->setBounds(bounds);
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker([&barrier](const ob::State *state)
                                    { return barrier.isSafe(configurationOf(state)); });
        si->setStateValidityCheckingResolution(0.005);
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
        for (int j = 0; j < dimension; ++j)
        {
            start[j] = startConfiguration()[j];
            goal[j] = goalConfiguration()[j];
        }
        auto definition = std::make_shared<ob::ProblemDefinition>(si);
        definition->setStartAndGoalStates(start, goal);
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(2.0);
        planner->setProblemDefinition(definition);
        planner->setup();
        if (!planner->solve(ob::timedPlannerTerminationCondition(5.0)) || !definition->hasExactSolution())
            throw std::runtime_error("straight-edge RRTConnect did not solve the teaching scene");

        auto path = std::static_pointer_cast<og::PathGeometric>(definition->getSolutionPath());
        for (std::size_t i = 0; i < path->getStateCount(); ++i)
            nodes.push_back(configurationOf(path->getState(i)));
        const unsigned int samples = std::max(2u, static_cast<unsigned int>(path->length() / 0.02));
        path->interpolate(samples);
        std::vector<UR5::Configuration> dense;
        for (std::size_t i = 0; i < path->getStateCount(); ++i)
            dense.push_back(configurationOf(path->getState(i)));
        return dense;
    }

    std::vector<UR5::Configuration> cbfPath(const ompl::sdf::GridSDF &field,
                                            std::vector<UR5::Configuration> &nodes,
                                            std::size_t &filtered, std::size_t &misses)
    {
        const UR5 robot;
        const Barrier guard = Barrier::guarding(robot, field, Barrier::defaultMargin);
        const Filter filter(guard);
        auto space = std::make_shared<ompl::cbf::FilteredStateSpace>(filter, stepSize,
                                                                    UR5::velocityLimits());
        ob::RealVectorBounds bounds(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, UR5::lowerBounds()[j]);
            bounds.setHigh(j, UR5::upperBounds()[j]);
        }
        space->setBounds(bounds);
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::FilteredMotionValidator>(si));
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
        for (int j = 0; j < dimension; ++j)
        {
            start[j] = startConfiguration()[j];
            goal[j] = goalConfiguration()[j];
        }
        auto definition = std::make_shared<ob::ProblemDefinition>(si);
        definition->setStartAndGoalStates(start, goal, 0.35);
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(2.0);
        planner->setProblemDefinition(definition);
        planner->setup();
        if (!planner->solve(ob::timedPlannerTerminationCondition(5.0)) || !definition->hasExactSolution())
            throw std::runtime_error("CBF-RRTConnect did not solve the teaching scene");

        auto path = std::static_pointer_cast<og::PathGeometric>(definition->getSolutionPath());
        for (std::size_t i = 0; i < path->getStateCount(); ++i)
            nodes.push_back(ompl::cbf::FilteredStateSpace::configurationOf(path->getState(i)));
        const og::PathGeometric executed = ompl::cbf::executedPath(*path, 0.01, &misses);
        std::vector<UR5::Configuration> dense;
        for (std::size_t i = 0; i < executed.getStateCount(); ++i)
            dense.push_back(ompl::cbf::FilteredStateSpace::configurationOf(executed.getState(i)));
        filtered = space->statistics().filtered;
        return dense;
    }

    void writeConfigurations(std::FILE *out, const char *name, const UR5 &robot,
                             const std::vector<UR5::Configuration> &path)
    {
        std::fprintf(out, "  \"%s\": [\n", name);
        for (std::size_t s = 0; s < path.size(); ++s)
        {
            const auto centers = robot.sphereCenters(path[s]);
            std::fprintf(out, "    {\"q\": [");
            for (int j = 0; j < dimension; ++j)
                std::fprintf(out, "%s%.9g", j ? ", " : "", path[s][j]);
            std::fprintf(out, "], \"centers\": [");
            for (Eigen::Index i = 0; i < centers.cols(); ++i)
                std::fprintf(out, "%s[%.9g, %.9g, %.9g]", i ? ", " : "", centers(0, i), centers(1, i),
                             centers(2, i));
            std::fprintf(out, "]}%s\n", s + 1 < path.size() ? "," : "");
        }
        std::fprintf(out, "  ]");
    }

    void writeJson(const std::string &path, const std::vector<UR5::Configuration> &rrtNodes,
                   const std::vector<UR5::Configuration> &rrtPath,
                   const std::vector<UR5::Configuration> &cbfNodes,
                   const std::vector<UR5::Configuration> &cbfPath)
    {
        std::FILE *out = std::fopen(path.c_str(), "w");
        if (!out)
            throw std::runtime_error("cannot open output " + path);
        const UR5 robot;
        std::fprintf(out, "{\n  \"radii\": [");
        for (std::size_t i = 0; i < UR5::nSpheres; ++i)
            std::fprintf(out, "%s%.9g", i ? ", " : "", UR5::spheres()[i].radius);
        std::fprintf(out, "],\n  \"obstacles\": [");
        for (std::size_t i = 0; i < obstacleCenters.size(); ++i)
            std::fprintf(out, "%s{\"center\": [%.9g, %.9g, %.9g], \"radius\": %.9g}", i ? ", " : "",
                         obstacleCenters[i].x(), obstacleCenters[i].y(), obstacleCenters[i].z(), obstacleRadii[i]);
        std::fprintf(out, "],\n");
        writeConfigurations(out, "rrt_nodes", robot, rrtNodes);
        std::fprintf(out, ",\n");
        writeConfigurations(out, "rrt_path", robot, rrtPath);
        std::fprintf(out, ",\n");
        writeConfigurations(out, "cbf_nodes", robot, cbfNodes);
        std::fprintf(out, ",\n");
        writeConfigurations(out, "cbf_path", robot, cbfPath);
        std::fprintf(out, "\n}\n");
        std::fclose(out);
    }
}

int main(int argc, char **argv)
{
    const std::string output = argc > 1 ? argv[1] : "ur5_cbf_curve.json";
    ompl::RNG::setSeed(1);
    ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
    std::printf("Baking synthetic UR5 obstacle field...\n");
    const ompl::sdf::GridSDF field(scene(), UR5::reachableBounds(), 0.03);
    std::vector<UR5::Configuration> nodes;
    const auto rrt = baselinePath(field, nodes);
    std::vector<UR5::Configuration> cbfNodes;
    std::size_t filtered = 0;
    std::size_t misses = 0;
    const auto cbf = cbfPath(field, cbfNodes, filtered, misses);
    writeJson(output, nodes, rrt, cbfNodes, cbf);
    std::printf("RRTConnect: %zu sparse nodes, %zu rendered states\n", nodes.size(), rrt.size());
    std::printf("CBF-RRTConnect: %zu sparse nodes, %zu executed states, %zu filtered controls, %zu replay misses\n",
                cbfNodes.size(), cbf.size(), filtered, misses);
    std::printf("wrote %s\n", output.c_str());
    return 0;
}
