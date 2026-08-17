// Coupled 14-DoF Reachy2 planning with CBF-filtered OMPL propagation.
//
// The robot geometry/kinematics in ompl/robots/Reachy2.h is generated from the
// supplied spherized URDF and SRDF by scripts/generate_reachy2_model.py.
//
// Usage:
//   demo_Reachy2CBFPlanning [seconds] [out.path]
//       [left_x left_y left_z right_x right_y right_z] [shortcut_radians]
//   demo_Reachy2CBFPlanning [seconds] [out.path] [shortcut_radians]

// The output is one raw-radian configuration per row, in Reachy2::jointNames()
// order. By default both hands move into the shelf's lower bay.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <qpmad/solver.h>

#include <ompl/base/ScopedState.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/JointSteeringControlSampler.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/control/PathControl.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/robots/Reachy2.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace oc = ompl::control;
using Robot = ompl::robots::Reachy2;
using Configuration = Robot::Configuration;
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

    class ReachyBarrier
    {
    public:
        // The Reachy model was fitted directly as collision spheres; reserve
        // 10 mm for grid/interpolation and finite-step error. This is separate
        // from the UR5's 60 mm mesh-undercoverage margin.
        static constexpr double worldMargin = 0.010;
        static constexpr double selfMargin = 0.005;

        struct Evaluation
        {
            Eigen::VectorXd h;
            Eigen::MatrixXd rows;
            bool inBounds{true};
            double worstWorld{std::numeric_limits<double>::infinity()};
            double worstSelf{std::numeric_limits<double>::infinity()};
        };

        ReachyBarrier(const Robot &robot, const ompl::sdf::GridSDF &field,
                      const Configuration &reference) : robot_(robot), field_(field)
        {
            // Some non-disabled mesh pairs overlap in the sphere approximation at
            // the declared start pose. A CBF cannot make an initially negative invariant
            // true. Keep every semantic pair that is separated in the declared
            // start pose; the startup report makes this calibration explicit.
            const auto kin = robot_.kinematics(reference);
            for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
            {
                const auto pair = Robot::selfPairs()[p];
                const auto &sa = Robot::spheres()[pair.a];
                const auto &sb = Robot::spheres()[pair.b];
                const double gap = (Robot::sphereCenter(kin, pair.a) - Robot::sphereCenter(kin, pair.b)).norm() -
                                   sa.radius - sb.radius;
                // A tiny positive gap at one pose is not a usable invariant for
                // a coarse fitted sphere model. Reserve 2 cm of calibration
                // headroom; path audits still evaluate every retained pair.
                if (gap > selfMargin + 0.02)
                    selfPairs_.push_back(p);
            }
        }

        std::size_t constraintCount() const
        {
            return Robot::nSpheres + selfPairs_.size();
        }

        Evaluation evaluate(const Configuration &q, double activation = std::numeric_limits<double>::infinity()) const
        {
            const auto kin = robot_.kinematics(q);
            std::array<Eigen::Vector3d, Robot::nSpheres> centers;
            std::vector<double> values;
            std::vector<Eigen::Matrix<double, 1, N>> rows;
            values.reserve(constraintCount());
            rows.reserve(constraintCount());
            Evaluation out;

            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers[i] = Robot::sphereCenter(kin, i);
                out.inBounds = out.inBounds && field_.inBounds(centers[i]);
                const double h = field_.distance(centers[i]) - Robot::spheres()[i].radius - worldMargin;
                out.worstWorld = std::min(out.worstWorld, h);
                if (h <= activation)
                {
                    values.push_back(h);
                    rows.push_back(field_.gradient(centers[i]).transpose() * Robot::sphereJacobian(kin, i));
                }
            }

            for (const std::size_t p : selfPairs_)
            {
                const auto pair = Robot::selfPairs()[p];
                const Eigen::Vector3d delta = centers[pair.a] - centers[pair.b];
                const double distance = delta.norm();
                const double h = distance - Robot::spheres()[pair.a].radius -
                                 Robot::spheres()[pair.b].radius - selfMargin;
                out.worstSelf = std::min(out.worstSelf, h);
                if (h <= activation)
                {
                    values.push_back(h);
                    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
                    if (distance > 1e-12)
                        normal = delta / distance;
                    rows.push_back(normal.transpose() *
                                   (Robot::sphereJacobian(kin, pair.a) - Robot::sphereJacobian(kin, pair.b)));
                }
            }

            out.h.resize(values.size());
            out.rows.resize(values.size(), N);
            for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(values.size()); ++i)
            {
                out.h[i] = values[i];
                out.rows.row(i) = rows[i];
            }
            return out;
        }

        bool safe(const Configuration &q, double *world = nullptr, double *self = nullptr) const
        {
            const auto e = evaluate(q);
            if (world)
                *world = e.worstWorld;
            if (self)
                *self = e.worstSelf;
            return e.inBounds && e.worstWorld >= 0.0 && e.worstSelf >= 0.0;
        }

        std::size_t enabledSelfPairs() const
        {
            return selfPairs_.size();
        }

    private:
        const Robot &robot_;
        const ompl::sdf::GridSDF &field_;
        std::vector<std::size_t> selfPairs_;
    };

    class ReachyFilter
    {
    public:
        explicit ReachyFilter(const ReachyBarrier &barrier) : barrier_(barrier)
        {
        }

        bool filter(const Configuration &q, const Configuration &nominal, double dt,
                    Configuration &filtered) const
        {
            constexpr double gamma = 0.6;
            constexpr double activation = 0.12;
            const auto e = barrier_.evaluate(q, activation);
            if (!e.inBounds)
            {
                filtered.setZero();
                return false;
            }

            const Configuration speed = Robot::velocityLimits().cwiseMin(Configuration::Constant(1.2));
            Configuration lower = -speed;
            Configuration upper = speed;
            const Configuration qlo = Robot::lowerBounds();
            const Configuration qhi = Robot::upperBounds();
            for (int j = 0; j < N; ++j)
            {
                lower[j] = std::max(lower[j], (qlo[j] - q[j]) / dt);
                upper[j] = std::min(upper[j], (qhi[j] - q[j]) / dt);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }

            if (e.h.size() == 0)
            {
                filtered = nominal.cwiseMax(lower).cwiseMin(upper);
                return true;
            }

            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(N, N);
            Eigen::VectorXd objective = -nominal;
            Eigen::VectorXd rowLower = -gamma * e.h / dt;
            Eigen::VectorXd rowUpper = Eigen::VectorXd::Constant(e.h.size(),
                                                                 std::numeric_limits<double>::infinity());
            try
            {
                const auto status = solver_.solve(filtered, H, objective, lower, upper,
                                                  e.rows, rowLower, rowUpper);
                if (status == qpmad::Solver::OK)
                    return true;
            }
            catch (const std::exception &)
            {
            }
            filtered.setZero();
            return false;
        }

    private:
        const ReachyBarrier &barrier_;
        mutable qpmad::Solver solver_;
    };

    class Propagator : public oc::StatePropagator
    {
    public:
        Propagator(const oc::SpaceInformationPtr &si, const ReachyFilter &filter)
          : oc::StatePropagator(si), filter_(filter)
        {
        }

        void propagate(const ob::State *state, const oc::Control *control, double duration,
                       ob::State *result) const override
        {
            const double *src = state->as<ob::RealVectorStateSpace::StateType>()->values;
            const double *u = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            Configuration q, nominal;
            for (int j = 0; j < N; ++j)
            {
                q[j] = src[j];
                nominal[j] = u[j];
            }
            constexpr double integrationStep = 0.02;
            double elapsed = 0.0;
            while (elapsed < duration - 1e-12)
            {
                const double dt = std::min(integrationStep, duration - elapsed);
                Configuration applied;
                if (!filter_.filter(q, nominal, dt, applied))
                    break;
                q += applied * dt;
                elapsed += dt;
            }
            double *dst = result->as<ob::RealVectorStateSpace::StateType>()->values;
            for (int j = 0; j < N; ++j)
                dst[j] = q[j];
        }

        bool canPropagateBackward() const override
        {
            return false;
        }

    private:
        const ReachyFilter &filter_;
    };

    bool solveIK(const Robot &robot, const ReachyBarrier &barrier, const Eigen::Vector3d &left,
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

    void appendDistinct(std::vector<Configuration> &motion, const Configuration &q)
    {
        if (motion.empty() || (motion.back() - q).norm() > 1e-12)
            motion.push_back(q);
    }

    struct Rollout
    {
        std::vector<Configuration> waypoints;
        bool completed{false};
    };

    // Re-aim at the requested endpoint after every CBF step, matching the rollout
    // used by the repository's FilteredStateSpace shortcutter. Every returned
    // segment is an actually filtered constant-control integration step.
    Rollout rollToward(const ReachyFilter &filter, const Configuration &from,
                       const Configuration &to, const Configuration &speed)
    {
        Rollout out;
        out.waypoints.push_back(from);
        const double required = ((to - from).cwiseAbs().cwiseQuotient(speed)).maxCoeff();
        const unsigned int steps = std::max(1u, static_cast<unsigned int>(std::ceil(required / integrationStep)));
        const double horizon = steps * integrationStep;
        Configuration q = from;

        for (unsigned int step = 0; step < steps; ++step)
        {
            const double remaining = horizon - step * integrationStep;
            Configuration nominal;
            for (int j = 0; j < N; ++j)
                nominal[j] = std::clamp((to[j] - q[j]) / remaining, -speed[j], speed[j]);
            Configuration applied;
            if (!filter.filter(q, nominal, integrationStep, applied))
                return out;
            Configuration landing = q + applied * integrationStep;
            if ((landing - to).cwiseAbs().maxCoeff() <= 1e-12)
                landing = to;
            appendDistinct(out.waypoints, landing);
            q = landing;
        }
        out.completed = true;
        return out;
    }

    std::vector<Configuration> executedMotion(const oc::PathControl &path,
                                               const ReachyFilter &filter)
    {
        std::vector<Configuration> motion;
        if (path.getStateCount() == 0)
            return motion;
        motion.push_back(stateConfiguration(path.getState(0)));
        for (std::size_t i = 0; i < path.getControlCount(); ++i)
        {
            const double *raw = path.getControl(i)->as<oc::RealVectorControlSpace::ControlType>()->values;
            Configuration nominal;
            for (int j = 0; j < N; ++j)
                nominal[j] = raw[j];

            Configuration q = motion.back();
            double elapsed = 0.0;
            const double duration = path.getControlDuration(i);
            while (elapsed < duration - 1e-12)
            {
                const double dt = std::min(integrationStep, duration - elapsed);
                Configuration applied;
                if (!filter.filter(q, nominal, dt, applied))
                    break;
                q += applied * dt;
                appendDistinct(motion, q);
                elapsed += dt;
            }
            const Configuration recorded = stateConfiguration(path.getState(i + 1));
            if ((q - recorded).norm() > 1e-9)
                throw std::runtime_error("could not replay a planned Reachy control edge");
            motion.back() = recorded;
        }
        return motion;
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
    constexpr double shortcutEquivalenceTolerance = 0.01;
    // Default: both hands enter the lower bay. Both targets can be replaced by
    // the six CLI coordinates.
    Eigen::Vector3d left(0.62, 0.20, 1.1344), right(0.62, -0.20, 1.1344);
    if (argc == 9 || argc == 10)
    {
        for (int i = 0; i < 3; ++i)
            left[i] = std::stod(argv[3 + i]);
        for (int i = 0; i < 3; ++i)
            right[i] = std::stod(argv[6 + i]);
        if (argc == 10)
            shortcutDelta = std::stod(argv[9]);
    }
    else if (argc == 4)
        shortcutDelta = std::stod(argv[3]);
    else if (argc != 1 && argc != 2 && argc != 3)
    {
        std::fprintf(stderr,
                     "usage: %s [seconds] [out.path] [lx ly lz rx ry rz] [shortcutRadians]\n",
                     argv[0]);
        return 2;
    }
    if (shortcutDelta < 0.0)
    {
        std::fprintf(stderr, "shortcutRadians must be non-negative (zero disables it)\n");
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
    const ReachyBarrier startSearchBarrier(robot, field, reference);
    Configuration start;
    if (!solveIK(robot, startSearchBarrier, Eigen::Vector3d(0.25, 0.20, 1.1344),
                 Eigen::Vector3d(0.25, -0.20, 1.1344), start))
    {
        std::fprintf(stderr, "could not construct the safe pre-shelf start pose\n");
        return 3;
    }
    const ReachyBarrier barrier(robot, field, start);
    const ReachyFilter filter(barrier);

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

    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(N);
    ob::RealVectorBounds stateBounds(N);
    const Configuration lo = Robot::lowerBounds(), hi = Robot::upperBounds();
    for (int j = 0; j < N; ++j)
    {
        stateBounds.setLow(j, lo[j]);
        stateBounds.setHigh(j, hi[j]);
    }
    stateSpace->setBounds(stateBounds);
    auto controlSpace = std::make_shared<oc::RealVectorControlSpace>(stateSpace, N);
    ob::RealVectorBounds controlBounds(N);
    const Configuration speed = Robot::velocityLimits().cwiseMin(Configuration::Constant(1.2));
    for (int j = 0; j < N; ++j)
    {
        controlBounds.setLow(j, -speed[j]);
        controlBounds.setHigh(j, speed[j]);
    }
    controlSpace->setBounds(controlBounds);

    oc::SimpleSetup setup(controlSpace);
    const auto si = setup.getSpaceInformation();
    si->setPropagationStepSize(0.08);
    si->setMinMaxControlDuration(2, 50);
    si->setStatePropagator(std::make_shared<Propagator>(si, filter));
    si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
    si->setDirectedControlSamplerAllocator([](const oc::SpaceInformation *s)
                                           { return std::make_shared<ompl::cbf::JointSteeringControlSampler>(s); });

    ob::ScopedState<ob::RealVectorStateSpace> startState(stateSpace), goalState(stateSpace);
    for (int j = 0; j < N; ++j)
    {
        startState[j] = start[j];
        goalState[j] = goal[j];
    }
    // The public goal is Cartesian. A 0.15 rad joint-space tolerance can still leave
    // Reachy's long arm more than 4 cm from the requested tip position, despite OMPL
    // reporting an exact solution. Keep the planner's joint representative tight enough
    // for the Cartesian postcondition checked below.
    setup.setStartAndGoalStates(startState, goalState, 0.05);
    auto planner = std::make_shared<oc::RRT>(si);
    planner->setGoalBias(0.2);
    setup.setPlanner(planner);

    if (!setup.solve(seconds))
    {
        std::fprintf(stderr, "no CBF-safe plan found in %.1f seconds\n", seconds);
        return 5;
    }

    const auto &path = setup.getSolutionPath();
    std::vector<Configuration> motion = executedMotion(path, filter);
    ompl::cbf::ShortcutReport shortcut;
    double shortcutSeconds = 0.0;
    if (shortcutDelta > 0.0)
    {
        const auto begin = std::chrono::steady_clock::now();
        const double arrivalTolerance = 0.05 * speed.norm() * integrationStep;
        const auto rollout = [&filter, &speed](const Configuration &from,
                                               const Configuration &to)
        {
            Rollout result = rollToward(filter, from, to, speed);
            return result.completed ? std::move(result.waypoints)
                                    : std::vector<Configuration>();
        };
        motion = ompl::cbf::ropeShortcut(motion, shortcutDelta, arrivalTolerance,
                                         rollout, &shortcut, shortcutEquivalenceTolerance);
        shortcutSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("CBF shortcut: %.4f -> %.4f rad, %zu accepted from %zu rollouts, "
                    "%zu screened, %.1f ms, max arrival gap %.3f mm\n",
                    shortcut.lengthBefore, shortcut.lengthAfter, shortcut.accepted,
                    shortcut.rollouts, shortcut.screened, 1e3 * shortcutSeconds,
                    1e3 * shortcut.maxArrivalGap);
    }
    std::ofstream out(output);
    out.precision(17);
    out << "# joints";
    for (const char *name : Robot::jointNames())
        out << ' ' << name;
    out << '\n';
    double worstWorld = std::numeric_limits<double>::infinity();
    double worstSelf = std::numeric_limits<double>::infinity();
    for (const Configuration &q : motion)
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
        for (int j = 0; j < N; ++j)
            out << (j ? " " : "") << q[j];
        out << '\n';
    }
    std::printf("Solved: %zu audited states, worst world=%+.4f m, self=%+.4f m\n",
                motion.size(), worstWorld, worstSelf);
    const auto endKin = robot.kinematics(motion.back());
    const Eigen::Vector3d endLeft = Robot::tipPosition(endKin, true);
    const Eigen::Vector3d endRight = Robot::tipPosition(endKin, false);
    const double leftError = (endLeft - left).norm();
    const double rightError = (endRight - right).norm();
    std::printf("Final tips: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n",
                endLeft.x(), endLeft.y(), endLeft.z(), endRight.x(), endRight.y(), endRight.z());
    std::printf("Cartesian errors: left %.4f m, right %.4f m (%s joint-space solution)\n",
                leftError, rightError, setup.haveExactSolutionPath() ? "exact" : "approximate");
    // The requested goal is Cartesian; an approximate connection to one IK
    // representative is still a valid answer when both actual tips reach it.
    if (leftError > 0.04 || rightError > 0.04)
    {
        std::fprintf(stderr, "solution did not reach both Cartesian goals within 4 cm\n");
        return 7;
    }
    std::printf("Path: %s\n", output.c_str());
    return 0;
}
