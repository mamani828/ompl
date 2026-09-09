// Geometric RRT comparison for a holonomic point robot and staged OMPL optimizations:
//   qp_lipschitz: project the nominal velocity with the CBF QP and use the
//                 interval certificate to skip redundant filter calls.
//   qp_free_gate: keep the nominal velocity only while it satisfies the same
//                 CBF rows; truncate otherwise (the QP-free idea in
//                 LQR-CBF-RRT*, reduced to point-robot geometric planning).
//
// Both modes use the same OMPL RRT implementation, samples, state space bounds,
// range, goal bias, SDF, barrier, integration step, and termination condition.

#include <ompl/base/PlannerData.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/samplers/UniformValidStateSampler.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/RobotCBFControlFilter.h>
#include <ompl/cbf/RobotClearanceBarrier.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>
#include <ompl/util/Console.h>
#include <ompl/util/Time.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace
{
constexpr double kStep = 0.02;
constexpr double kRange = 0.75;
constexpr double kGoalBias = 0.05;
constexpr double kGoalTolerance = 0.12;
constexpr double kRobotRadius = 0.13;
// Covers the 4 cm SDF interpolation error plus sampled-control integration
// error. This value was increased until a 1,000-paired-seed audit was clean.
constexpr double kGuard = 0.03;
constexpr double kAuditResolution = 0.002;
constexpr std::array<double, 4> kBounds{-4.5, 4.5, -2.5, 2.5};

struct PointRobot
{
    // RobotClearanceBarrier models a mobile base as [x, y, yaw].  Yaw is fixed
    // to zero by the sampler and has no influence on the point sphere.
    static constexpr std::size_t nJoints = 3;
    static constexpr std::size_t nBaseJoints = 3;
    static constexpr std::size_t nSpheres = 1;
    // Disabled invariant sentinel; this avoids fixed-size zero-row Eigen types.
    static constexpr std::size_t nSelfPairs = 1;

    using Configuration = Eigen::Matrix<double, 3, 1>;
    using Kinematics = Configuration;
    using Jacobian = Eigen::Matrix<double, 3, 3>;

    struct Sphere
    {
        int link;
        std::array<double, 3> center;
        double radius;
        std::uint16_t influence;
        const char *name;
    };
    struct SelfPair
    {
        std::uint16_t a, b;
    };
    struct LinkStep
    {
        int parent;
        int active;
        std::array<double, 3> xyz;
    };

    static const std::array<LinkStep, 0> &steps()
    {
        static const std::array<LinkStep, 0> result{};
        return result;
    }
    static const std::array<Sphere, nSpheres> &spheres()
    {
        static const std::array<Sphere, nSpheres> result{{
            {0, {0.0, 0.0, 0.0}, kRobotRadius, 0x3, "point"},
        }};
        return result;
    }
    static const std::array<SelfPair, nSelfPairs> &selfPairs()
    {
        static const std::array<SelfPair, nSelfPairs> result{{{0, 0}}};
        return result;
    }
    static Configuration lowerBounds()
    {
        return (Configuration() << kBounds[0], kBounds[2], -1e-6).finished();
    }
    static Configuration upperBounds()
    {
        return (Configuration() << kBounds[1], kBounds[3], 1e-6).finished();
    }
    static Configuration velocityLimits()
    {
        return (Configuration() << 1.0, 1.0, 0.01).finished();
    }
    Kinematics kinematics(const Configuration &q) const
    {
        return q;
    }
    static Eigen::Vector3d sphereCenter(const Kinematics &q, std::size_t)
    {
        return {q[0], q[1], 0.0};
    }
    static Jacobian sphereJacobian(const Kinematics &, std::size_t)
    {
        Jacobian result = Jacobian::Zero();
        result(0, 0) = 1.0;
        result(1, 1) = 1.0;
        return result;
    }
};

using Barrier = ompl::cbf::RobotClearanceBarrier<PointRobot>;
using QPFilter = ompl::cbf::RobotCBFControlFilter<PointRobot>;
using Space = ompl::cbf::RobotFilteredStateSpace<PointRobot>;

struct Obstacle
{
    Eigen::Vector2d center;
    double radius;
};

const std::array<Obstacle, 3> kObstacles{{
    {Eigen::Vector2d(-1.55, -0.25), 0.72},
    {Eigen::Vector2d(0.00, 0.72), 0.82},
    {Eigen::Vector2d(1.55, -0.38), 0.70},
}};

ompl::sdf::DistanceFn scene()
{
    return [](const Eigen::Vector3d &p)
    {
        double distance = std::numeric_limits<double>::infinity();
        for (const auto &obstacle : kObstacles)
            distance = std::min(distance, (p.head<2>() - obstacle.center).norm() - obstacle.radius);
        return distance;
    };
}

// QP-free CBF steering from LQR-CBF-RRT*: accept the nominal input if every
// CBF inequality holds, otherwise stop the rollout.  For a holonomic point
// robot the LQR nominal is collinear with the geometric steer-to-sample input,
// so retaining an LQR gain would only reparameterize time.
class QPFreeGate final : public ompl::cbf::RobotControlFilter<PointRobot>
{
public:
    explicit QPFreeGate(const Barrier &barrier) : barrier_(barrier)
    {
        threshold_.setConstant(std::numeric_limits<double>::infinity());
    }

    Status filter(const Configuration &q, const Control &nominal, double duration,
                  Control &applied) const override
    {
        ++calls_;
        applied.setZero();
        if (duration <= 0.0)
        {
            ++rejected_;
            return Status::Blocked;
        }

        barrier_.evaluateScreened(q, threshold_, evaluation_);
        rows_ += static_cast<std::size_t>(evaluation_.active);
        if (!evaluation_.inBounds)
        {
            ++rejected_;
            return Status::Blocked;
        }

        for (Eigen::Index row = 0; row < evaluation_.active; ++row)
        {
            const auto constraint = evaluation_.constraint[row];
            const double h = evaluation_.values[constraint] - kGuard;
            const double cbf = evaluation_.rows.row(row).dot(nominal) + QPFilter::kappa * h;
            if (h < 0.0 || cbf < 0.0)
            {
                ++rejected_;
                return Status::Blocked;
            }
        }

        applied = nominal;
        return Status::Unchanged;
    }

    const char *name() const override
    {
        return "qp-free-cbf-gate";
    }
    std::size_t calls() const
    {
        return calls_;
    }
    std::size_t rejected() const
    {
        return rejected_;
    }
    double meanRows() const
    {
        return calls_ ? static_cast<double>(rows_) / static_cast<double>(calls_) : 0.0;
    }

private:
    const Barrier &barrier_;
    mutable typename Barrier::Values threshold_;
    mutable typename Barrier::Evaluation evaluation_;
    mutable std::size_t calls_{0};
    mutable std::size_t rejected_{0};
    mutable std::size_t rows_{0};
};

class SeededPointSampler final : public ob::RealVectorStateSampler
{
public:
    SeededPointSampler(const ob::StateSpace *space, std::uint_fast32_t seed)
      : ob::RealVectorStateSampler(space)
    {
        rng_.setLocalSeed(seed);
    }

    void sampleUniform(ob::State *state) override
    {
        ob::RealVectorStateSampler::sampleUniform(state);
        state->as<ob::RealVectorStateSpace::StateType>()->values[2] = 0.0;
    }
    void sampleUniformNear(ob::State *state, const ob::State *near, double distance) override
    {
        ob::RealVectorStateSampler::sampleUniformNear(state, near, distance);
        state->as<ob::RealVectorStateSpace::StateType>()->values[2] = 0.0;
    }
    void sampleGaussian(ob::State *state, const ob::State *mean, double stdDev) override
    {
        ob::RealVectorStateSampler::sampleGaussian(state, mean, stdDev);
        state->as<ob::RealVectorStateSpace::StateType>()->values[2] = 0.0;
    }
};

class SeededRRT final : public og::RRT
{
public:
    SeededRRT(const ob::SpaceInformationPtr &si, std::uint_fast32_t seed) : og::RRT(si)
    {
        rng_.setLocalSeed(seed);
    }
};

struct Result
{
    bool solved{false};
    double milliseconds{0.0};
    std::size_t vertices{0};
    std::size_t rollouts{0};
    std::size_t filterCalls{0};
    std::size_t qpCalls{0};
    std::size_t rejected{0};
    std::size_t coarseHops{0};
    double meanRows{0.0};
    double pathLength{std::numeric_limits<double>::quiet_NaN()};
    double minimumClearance{std::numeric_limits<double>::quiet_NaN()};
    std::size_t unsafeAuditSamples{0};
    std::size_t replayMisses{0};
};

PointRobot::Configuration configuration(const ob::State *state)
{
    return Eigen::Map<const PointRobot::Configuration>(
        state->as<ob::RealVectorStateSpace::StateType>()->values);
}

double exactBarrierClearance(const PointRobot::Configuration &q)
{
    double clearance = std::numeric_limits<double>::infinity();
    for (const auto &obstacle : kObstacles)
        clearance = std::min(clearance,
                             (q.head<2>() - obstacle.center).norm() - obstacle.radius -
                                 kRobotRadius - Barrier::worldMargin);
    return clearance;
}

void audit(const og::PathGeometric &path, Result &result)
{
    result.pathLength = 0.0;
    result.minimumClearance = std::numeric_limits<double>::infinity();
    for (std::size_t edge = 0; edge + 1 < path.getStateCount(); ++edge)
    {
        const auto a = configuration(path.getState(edge));
        const auto b = configuration(path.getState(edge + 1));
        const double length = (b.head<2>() - a.head<2>()).norm();
        result.pathLength += length;
        const std::size_t samples = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(length / kAuditResolution)));
        for (std::size_t i = 0; i <= samples; ++i)
        {
            const auto q = a + (static_cast<double>(i) / static_cast<double>(samples)) * (b - a);
            const double clearance = exactBarrierClearance(q);
            result.minimumClearance = std::min(result.minimumClearance, clearance);
            if (clearance < 0.0)
                ++result.unsafeAuditSamples;
        }
    }
}

template <typename Filter>
Result runOne(const char *mode, const Barrier &barrier, Filter &filter,
              std::uint_fast32_t seed, unsigned int iterations)
{
    auto space = std::make_shared<Space>(filter, kStep, PointRobot::velocityLimits());
    space->setStateSamplerAllocator([seed](const ob::StateSpace *stateSpace)
                                    { return std::make_shared<SeededPointSampler>(stateSpace, seed); });
    // qp_lipschitz spends sound interval certificates; QP-free deliberately
    // reports no certificate and therefore evaluates its inequality every step.
    space->setMaxStepScale(std::numeric_limits<double>::infinity());

    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
    si->setMotionValidator(std::make_shared<ompl::cbf::RobotFilteredMotionValidator<PointRobot>>(si));
    si->setup();

    ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
    start[0] = -4.0;
    start[1] = -1.35;
    start[2] = 0.0;
    goal[0] = 4.0;
    goal[1] = 1.35;
    goal[2] = 0.0;

    auto definition = std::make_shared<ob::ProblemDefinition>(si);
    definition->setStartAndGoalStates(start, goal, kGoalTolerance);
    auto planner = std::make_shared<SeededRRT>(si, seed + 1);
    planner->setRange(kRange);
    planner->setGoalBias(kGoalBias);
    planner->setProblemDefinition(definition);
    planner->setup();

    const auto begin = std::chrono::steady_clock::now();
    const auto status = planner->solve(ob::IterationTerminationCondition(iterations));
    const auto end = std::chrono::steady_clock::now();

    Result result;
    result.solved = status && definition->hasExactSolution();
    result.milliseconds = std::chrono::duration<double, std::milli>(end - begin).count();
    ob::PlannerData data(si);
    planner->getPlannerData(data);
    result.vertices = data.numVertices();
    const auto &statistics = space->statistics();
    result.rollouts = statistics.rollouts;
    result.coarseHops = statistics.coarse;

    if constexpr (std::is_same_v<Filter, QPFilter>)
    {
        result.filterCalls = filter.calls();
        result.qpCalls = filter.qpCalls();
        result.meanRows = filter.meanActiveRows();
    }
    else
    {
        result.filterCalls = filter.calls();
        result.rejected = filter.rejected();
        result.meanRows = filter.meanRows();
    }

    if (result.solved)
    {
        const auto sparse = std::static_pointer_cast<og::PathGeometric>(definition->getSolutionPath());
        const auto executed = ompl::cbf::robotExecutedPath<PointRobot>(*sparse, 0.0, &result.replayMisses);
        audit(executed, result);
    }
    (void)mode;
    (void)barrier;
    return result;
}

void printRow(std::uint_fast32_t seed, const char *mode, const Result &r)
{
    std::cout << seed << ',' << mode << ',' << (r.solved ? 1 : 0) << ','
              << std::setprecision(10) << r.milliseconds << ',' << r.vertices << ','
              << r.rollouts << ',' << r.filterCalls << ',' << r.qpCalls << ','
              << r.rejected << ',' << r.coarseHops << ',' << r.meanRows << ',';
    if (r.solved)
        std::cout << r.pathLength << ',' << r.minimumClearance;
    else
        std::cout << "nan,nan";
    std::cout << ',' << r.unsafeAuditSamples << ',' << r.replayMisses << '\n';
}
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
        // Individual point-robot queries are sub-millisecond, so a large paired
        // sample is the default to keep timer noise from dominating the result.
        const unsigned int trials = argc > 1 ? static_cast<unsigned int>(std::stoul(argv[1])) : 1000;
        const unsigned int iterations = argc > 2 ? static_cast<unsigned int>(std::stoul(argv[2])) : 5000;
        const std::uint_fast32_t firstSeed = argc > 3 ? std::stoul(argv[3]) : 17;
        if (trials == 0 || iterations == 0)
            throw std::invalid_argument("trials and iterations must be positive");

        const Eigen::AlignedBox3d fieldBounds(Eigen::Vector3d(-5.0, -3.0, -2.0),
                                              Eigen::Vector3d(5.0, 3.0, 2.0));
        // The robot is planar but GridSDF is 3-D; 4 cm keeps setup cheap while
        // the independent path audit below remains at 2 mm resolution.
        const ompl::sdf::GridSDF field(scene(), fieldBounds, 0.04);
        const PointRobot robot;

        std::cout << "seed,mode,solved,planning_ms,vertices,rollouts,filter_calls,qp_calls,"
                     "rejected,coarse_hops,mean_rows,path_length,min_clearance,unsafe_audit_samples,replay_misses\n";
        for (unsigned int trial = 0; trial < trials; ++trial)
        {
            const std::uint_fast32_t seed = firstSeed + trial;

            // Recreate stateful barriers/filters for every run. Alternating order
            // suppresses systematic warm-cache and frequency-scaling bias.
            auto runQP = [&]()
            {
                const Barrier barrier(robot, field, PointRobot::Configuration::Zero());
                QPFilter filter(barrier, PointRobot::lowerBounds(), PointRobot::upperBounds(), kGuard);
                printRow(seed, "qp_lipschitz", runOne("qp_lipschitz", barrier, filter, seed, iterations));
            };
            auto runGate = [&]()
            {
                const Barrier barrier(robot, field, PointRobot::Configuration::Zero());
                QPFreeGate filter(barrier);
                printRow(seed, "qp_free_gate", runOne("qp_free_gate", barrier, filter, seed, iterations));
            };

            if (trial % 2 == 0)
            {
                runQP();
                runGate();
            }
            else
            {
                runGate();
                runQP();
            }
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
