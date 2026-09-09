// demos/BaxterMBMBenchmark.cpp
//
// The UR5 MotionBenchMaker comparison, run on Baxter. Same scene format, same CSV
// schema, same two questions: how long a CBF rollout takes against ordinary
// collision-checked planning, and whether the motion it returns is actually clear.
//
//     python3 scripts/mbm_to_scenes.py external/vamp/resources/baxter/problems.json \
//         baxter_scenes.txt baxter
//     ./build/demos/demo_BaxterMBMBenchmark baxter_scenes.txt [perScene] [seconds] ...
//
// ### Why a second file rather than a template over the first
//
// `UR5MBMBenchmark.cpp` is bound to `cbf::ClearanceBarrier`, which is the hand-written
// UR5-only barrier with the baked self-pair table and its calibrated per-pair margins.
// Baxter goes through `cbf::RobotClearanceBarrier<Robot>`, which derives its lever-arm
// bounds and chooses its self-pairs at construction instead. Those are different enough
// that one file serving both would be a file with two of everything in it. What is
// shared is the part that has to be shared for the numbers to mean anything side by
// side: the scene reader, the audit resolution, and the CSV columns.
//
// ### What Baxter changes
//
// Fourteen joints and two arms that can reach each other. After the SRDF's allowed
// collisions and the rigid-body rule, 2061 sphere pairs remain, and the band rule that
// takes the UR5 from 780 to 303 only takes Baxter to 1680 -- those pairs are not slack,
// they are what a dual-arm robot is. So every barrier evaluation here carries roughly
// six times the UR5's rows, and the filter is correspondingly slower per call. The
// certificate ceiling moves the other way: no UR5 pair is ever clearer than 58 mm, while
// Baxter's is 399 mm, so its self-collision rows cap a certificate far less often.
//
// ### The rows
//
// - `isSafe`     RRTConnect over an ordinary collision checker. The baseline.
// - `cbfRRTC`    RRTConnect over `RobotFilteredStateSpace`: every edge is a QP-filtered
//                rollout, no state validity checker in the loop, and a hop may spend the
//                safety certificate (`RobotClearanceBarrier::safeDuration()`).
// - `cbfNoCert`  The same QP rollout with `maxStepScale = 1`, which collapses every hop
//                to one `stepSize`. The certificate can then never extend a hop, so this
//                row minus `cbfRRTC` is what the certificate buys and nothing else.
// - `qpFreeGate` The accept-or-stop CBF gate adapted from LQR-CBF-RRT* -- same rows, same
//                control box, but a failing row stops the rollout instead of being
//                repaired, and no certificate is issued. `demos/RobotQPFreeGate.h`.
// - `VAMP`       SIMD collision-checked RRTConnect, when built with VAMP.
//
// The `[maxStepScale]` and `[safeHops]` arguments configure the `cbfRRTC` row only; the
// other two are pinned by what they are measuring.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/base/PlannerData.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/RobotCBFControlFilter.h>
#include <ompl/cbf/RobotClearanceBarrier.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/robots/Baxter.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/Exception.h>

#include "RobotQPFreeGate.h"

#ifdef OMPL_MBM_HAVE_VAMP
#include <ompl/vamp/Utils.h>
#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/baxter.hh>
#endif

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace sdf = ompl::sdf;
using Robot = ompl::robots::Baxter;
using Barrier = ompl::cbf::RobotClearanceBarrier<Robot>;
using Filter = ompl::cbf::RobotCBFControlFilter<Robot>;
using Space = ompl::cbf::RobotFilteredStateSpace<Robot>;
using Configuration = Robot::Configuration;

namespace
{
    constexpr int dimension = static_cast<int>(Robot::nJoints);
    constexpr int checkedRow = 0;
    constexpr int cbfRow = 1;
    /// The same QP rollout with the hop certificate switched off -- one filter call per
    /// `stepSize`, which is what `maxStepScale = 1` collapses the span to. Isolates what
    /// the certificate buys from what the QP buys.
    constexpr int noCertRow = 2;
    /// Accept-or-stop CBF gate, no QP at all. See `demos/RobotQPFreeGate.h`.
    constexpr int gateRow = 3;
#ifdef OMPL_MBM_HAVE_VAMP
    constexpr int vampRow = 4;
    constexpr int comparisonRows = 5;
#else
    constexpr int comparisonRows = 4;
#endif

    /// Joint-space spacing every row is audited at, in radians. Matches the UR5 harness
    /// so the unsafe counts are comparable, and is finer than the rollout step so the
    /// audit is not re-reading the filter's own decisions.
    constexpr double auditResolution = 0.02;

    /// One obstacle: a box (`halfExtents`) or a cylinder (`radius`, `halfLength` about
    /// the local z axis), posed in the world. Identical to the UR5 harness's -- both
    /// read the same scene files, and every MotionBenchMaker obstacle is one of these
    /// two, so the field the barrier reads is exact up to the grid.
    struct Obstacle
    {
        enum class Kind
        {
            Box,
            Cylinder
        };

        Kind kind{Kind::Box};
        Eigen::Vector3d halfExtents{Eigen::Vector3d::Zero()};
        double radius{0.0};
        double halfLength{0.0};
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};

        double distance(const Eigen::Vector3d &p) const
        {
            const Eigen::Vector3d q = rotation.transpose() * (p - position);
            if (kind == Kind::Box)
            {
                const Eigen::Vector3d d = q.cwiseAbs() - halfExtents;
                return d.cwiseMax(0.0).norm() + std::min(d.maxCoeff(), 0.0);
            }
            const Eigen::Vector2d d(q.head<2>().norm() - radius, std::abs(q.z()) - halfLength);
            return d.cwiseMax(0.0).norm() + std::min(d.maxCoeff(), 0.0);
        }
    };

    struct Problem
    {
        std::string scene;
        int index{0};
        Configuration start{Configuration::Zero()};
        Configuration goal{Configuration::Zero()};
        std::vector<Obstacle> obstacles;

        /// A union of solids is the min of their distances, which is still exact.
        sdf::DistanceFn field() const
        {
            const std::vector<Obstacle> solids = obstacles;
            return [solids](const Eigen::Vector3d &p)
            {
                double distance = std::numeric_limits<double>::infinity();
                for (const Obstacle &solid : solids)
                    distance = std::min(distance, solid.distance(p));
                return distance;
            };
        }
    };

    Eigen::Matrix3d rotationOf(double x, double y, double z, double w)
    {
        return Eigen::Quaterniond(w, x, y, z).normalized().toRotationMatrix();
    }

    std::vector<Problem> readProblems(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw ompl::Exception("cannot open " + path +
                                  " -- generate it with scripts/mbm_to_scenes.py");

        std::vector<Problem> problems;
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream fields(line);
            std::string tag;
            fields >> tag;

            if (tag == "problem")
            {
                problems.emplace_back();
                fields >> problems.back().scene >> problems.back().index;
            }
            else if (problems.empty())
            {
                continue;
            }
            else if (tag == "start" || tag == "goal")
            {
                Configuration q;
                for (int j = 0; j < dimension; ++j)
                    fields >> q[j];
                if (!fields)
                    throw ompl::Exception("scene file has fewer than " +
                                          std::to_string(dimension) + " joints per configuration");
                (tag == "start" ? problems.back().start : problems.back().goal) = q;
            }
            else if (tag == "box" || tag == "cyl")
            {
                Obstacle obstacle;
                double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
                if (tag == "box")
                {
                    obstacle.kind = Obstacle::Kind::Box;
                    fields >> obstacle.halfExtents[0] >> obstacle.halfExtents[1] >>
                        obstacle.halfExtents[2];
                }
                else
                {
                    obstacle.kind = Obstacle::Kind::Cylinder;
                    fields >> obstacle.radius >> obstacle.halfLength;
                }
                fields >> obstacle.position[0] >> obstacle.position[1] >> obstacle.position[2] >>
                    qx >> qy >> qz >> qw;
                obstacle.rotation = rotationOf(qx, qy, qz, qw);
                problems.back().obstacles.push_back(obstacle);
            }
        }
        return problems;
    }

    struct Result
    {
        bool solved{false};
        double seconds{0.0};
        std::size_t evaluations{0};
        std::size_t vertices{0};
        std::size_t primarySamples{0};
        std::size_t productiveSamples{0};
        double pathLength{0.0};
        std::size_t waypoints{0};
        std::size_t unsafeStates{0};
        std::size_t auditedStates{0};
        std::size_t misses{0};
        double radPerCall{0.0};
        double coarse{0.0};
        double minClearance{std::numeric_limits<double>::infinity()};
        double minSelfOverlap{std::numeric_limits<double>::infinity()};
        std::size_t selfColliding{0};
        std::vector<Configuration> motion;
    };

    /// Deterministic sampling, so a row's seed is the only thing that varies between
    /// them. Both rows allocate their own sampler from the same seed.
    class SeededSampler : public ob::RealVectorStateSampler
    {
    public:
        SeededSampler(const ob::StateSpace *space, std::uint_fast32_t seed)
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

    /// The box the SDF is baked over. Measured, not guessed: over 200k configurations
    /// uniform in the joint limits, sphere *centres* span x[-1.08, 1.27], y[-1.46, 1.46],
    /// z[-0.71, 1.54]. Centres are what has to stay inside -- that is what `inBounds()`
    /// tests and where `GridSDF` clamps and over-reports -- so the sphere radii do not
    /// enter, and 8 cm of pad is several voxels at any resolution worth using.
    ///
    /// Sized deliberately tight. The grid is dense, so the box is the whole cost: this
    /// one is 2.5 x 3.1 x 2.4 m against the 3.6 x 4.1 x 3.5 m it started as, which is 2.7x
    /// fewer cells at the same voxel and is what makes a 10 mm grid affordable at all.
    /// Obstacles reaching outside it are still exact, since every node's distance is
    /// evaluated against the original primitives rather than against a clipped copy.
    Eigen::AlignedBox3d reachableBounds()
    {
        return Eigen::AlignedBox3d(Eigen::Vector3d(-1.16, -1.54, -0.79),
                                   Eigen::Vector3d(1.35, 1.54, 1.62));
    }

    ob::RealVectorBounds jointBounds()
    {
        ob::RealVectorBounds bounds(dimension);
        const auto lo = Robot::lowerBounds();
        const auto hi = Robot::upperBounds();
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, lo[j]);
            bounds.setHigh(j, hi[j]);
        }
        return bounds;
    }

    Configuration stateConfiguration(const ob::State *state)
    {
        const auto *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
        Configuration q;
        for (int j = 0; j < dimension; ++j)
            q[j] = values[j];
        return q;
    }

    /// Exact primitive clearance of every sphere at \p q, against the original boxes and
    /// cylinders rather than against the grid the planner read. This is the check, and it
    /// shares no code with the field the barrier queries.
    double exactClearance(const Robot &robot, const std::vector<Obstacle> &obstacles,
                          const Configuration &q)
    {
        const auto kin = robot.kinematics(q);
        double worst = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
        {
            const Eigen::Vector3d centre = Robot::sphereCenter(kin, i);
            double distance = std::numeric_limits<double>::infinity();
            for (const Obstacle &solid : obstacles)
                distance = std::min(distance, solid.distance(centre));
            worst = std::min(worst, distance - Robot::spheres()[i].radius);
        }
        return worst;
    }

    /// Every self-pair in the *generated* table, not merely the ones the barrier chose to
    /// enable. That is the point: it is the only thing able to say the enabling was wrong.
    double worstSelfOverlap(const Robot &robot, const Configuration &q)
    {
        const auto kin = robot.kinematics(q);
        std::vector<Eigen::Vector3d> centres(Robot::nSpheres);
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            centres[i] = Robot::sphereCenter(kin, i);
        double worst = std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
        {
            const auto pair = Robot::selfPairs()[p];
            const auto &a = Robot::spheres()[pair.a];
            const auto &b = Robot::spheres()[pair.b];
            if (a.influence == b.influence)  // rigidly linked: not a collision the planner can avoid
                continue;
            worst = std::min(worst, (centres[pair.a] - centres[pair.b]).norm() - a.radius - b.radius);
        }
        return worst;
    }

    /// The barrier's *own* reading along a motion it certified. If this goes negative the
    /// filter did not hold its invariant; if it stays non-negative while the exact audit
    /// goes negative, the field it reads is the thing at fault. The two answers point at
    /// completely different bugs, so it is worth the extra pass.
    double worstBarrierValue(const Barrier &barrier, const std::vector<Configuration> &motion)
    {
        double worst = std::numeric_limits<double>::infinity();
        // A zero threshold screens every row out, so this fills `values` for all of them
        // and builds no gradients -- the cheap call, and the one that reads the barrier
        // exactly as the filter's own evaluation would.
        typename Barrier::Values threshold = Barrier::Values::Zero();
        typename Barrier::Evaluation evaluation;
        for (const Configuration &q : motion)
        {
            barrier.evaluateScreened(q, threshold, evaluation);
            if (!evaluation.inBounds)
                return -std::numeric_limits<double>::infinity();
            worst = std::min(worst,
                             evaluation.values.head(static_cast<Eigen::Index>(barrier.constraintCount()))
                                 .minCoeff());
        }
        return worst;
    }

    void auditMotion(Result &result, const Robot &robot, const std::vector<Obstacle> &obstacles,
                     double margin)
    {
        for (const Configuration &q : result.motion)
        {
            ++result.auditedStates;
            // Net of the audited margin, matching UR5MBMBenchmark's column exactly: that
            // harness records `exactWorldClearance(..., barrier.margin())`, so a zero here
            // means "touching the margin", not "touching the obstacle". The two CSVs are
            // read side by side, and a column that means two things is worse than useless.
            const double h = exactClearance(robot, obstacles, q) - margin;
            result.minClearance = std::min(result.minClearance, h);
            if (h < 0.0)
                ++result.unsafeStates;
            const double overlap = worstSelfOverlap(robot, q);
            result.minSelfOverlap = std::min(result.minSelfOverlap, overlap);
            if (overlap < 0.0)
                ++result.selfColliding;
        }
    }

    double pathLength(const std::vector<Configuration> &motion)
    {
        double total = 0.0;
        for (std::size_t i = 1; i < motion.size(); ++i)
            total += (motion[i] - motion[i - 1]).norm();
        return total;
    }
}  // namespace

namespace
{
    /// Ordinary collision-checked RRTConnect over the same SDF the CBF rows read. The bar.
    Result runChecked(const Robot &robot, const sdf::GridSDF &field,
                      const std::vector<Obstacle> &obstacles, const Problem &problem,
                      double margin, double selfMargin, double range, double timeLimit,
                      double resolution, std::uint_fast32_t seed)
    {
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *s) { return std::make_shared<SeededSampler>(s, seed + 1); });
        auto si = std::make_shared<ob::SpaceInformation>(space);

        std::size_t checks = 0;
        si->setStateValidityChecker(
            [&robot, &field, margin, selfMargin, &checks](const ob::State *state)
            {
                ++checks;
                const Configuration q = stateConfiguration(state);
                const auto kin = robot.kinematics(q);
                for (std::size_t i = 0; i < Robot::nSpheres; ++i)
                {
                    const Eigen::Vector3d c = Robot::sphereCenter(kin, i);
                    if (!field.inBounds(c))
                        return false;
                    if (field.distance(c) - Robot::spheres()[i].radius - margin < 0.0)
                        return false;
                }
                // The arm against itself. The UR5 harness's baseline can leave this to the
                // audit, because a UR5 in a bookshelf hardly ever folds through itself. Two
                // Baxter arms in the same shelf do, constantly -- so a baseline without it
                // is not slower than the CBF row, it is solving a different problem.
                return worstSelfOverlap(robot, q) >= selfMargin;
            });
        si->setStateValidityCheckingResolution(resolution);
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> startState(space), goalState(space);
        for (int j = 0; j < dimension; ++j)
        {
            startState[j] = problem.start[j];
            goalState[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(startState, goalState, 0.05);
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(range);
        planner->setProblemDefinition(pdef);
        planner->setup();

        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        Result result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.evaluations = checks;
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();
        result.solved = status == ob::PlannerStatus::EXACT_SOLUTION && pdef->hasSolution();
        if (!result.solved)
            return result;

        auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        solution->interpolate(static_cast<unsigned int>(
            std::max(2.0, solution->length() / auditResolution)));
        result.waypoints = solution->getStateCount();
        for (std::size_t i = 0; i < solution->getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(solution->getState(i)));
        result.pathLength = pathLength(result.motion);
        auditMotion(result, robot, obstacles, margin);
        return result;
    }

    /// The same planner over `RobotFilteredStateSpace`, so every edge is a CBF rollout and
    /// every intermediate state is certified as it is produced. No state validity checker
    /// at all -- the filter is the only safety mechanism in the loop.
    ///
    /// \p filter is whichever local decision the row under test uses -- the QP filter or
    /// the accept-or-stop gate -- taken through the base interface so all of them run the
    /// identical planner, sampler, validator and audit. \p maxStepScale caps a hop at
    /// that multiple of \p stepSize; at 1 the certificate can never extend a hop, which
    /// is the "no certificate" row. \p safeHops picks which certificate a hop may spend --
    /// the safety one or the shorter no-op one; see `FilteredStateSpace::setSafeHops()`.
    Result runFiltered(const Robot &robot, const Barrier &barrier,
                       const typename Space::Filter &filter,
                       const std::vector<Obstacle> &obstacles, const Problem &problem,
                       double margin, double stepSize, double range, double timeLimit,
                       std::uint_fast32_t seed, double maxStepScale, bool safeHops)
    {
        auto space = std::make_shared<Space>(filter, stepSize, Filter::maxSpeed());
        typename Space::EarlyTermination earlyTermination;
        earlyTermination.enabled = true;
        space->setEarlyTermination(earlyTermination);
        space->setMaxStepScale(maxStepScale);
        space->setSafeHops(safeHops);
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *s) { return std::make_shared<SeededSampler>(s, seed + 1); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::RobotFilteredMotionValidator<Robot>>(si));
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> startState(space), goalState(space);
        for (int j = 0; j < dimension; ++j)
        {
            startState[j] = problem.start[j];
            goalState[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(startState, goalState, 0.05);
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(range);
        planner->setProblemDefinition(pdef);
        planner->setup();

        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        Result result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const auto stats = space->statistics();
        result.evaluations = stats.steps;
        result.radPerCall = stats.steps > 0 ? stats.travel / static_cast<double>(stats.steps) : 0.0;
        result.coarse = stats.steps > 0 ? static_cast<double>(stats.coarse) / stats.steps : 0.0;
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();
        result.solved = status == ob::PlannerStatus::EXACT_SOLUTION && pdef->hasSolution();
        if (!result.solved)
            return result;

        // The rollout the planner actually recorded for each edge, replayed -- so the audit
        // reads the motion that would be executed, not a straight line between waypoints.
        const auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        const og::PathGeometric executed =
            ompl::cbf::robotExecutedPath<Robot>(*solution, auditResolution, &result.misses);
        result.waypoints = executed.getStateCount();
        for (std::size_t i = 0; i < executed.getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(executed.getState(i)));
        result.pathLength = pathLength(result.motion);
        auditMotion(result, robot, obstacles, margin);
        if (std::getenv("OMPL_CBF_DIAG") != nullptr)
            std::printf("    [diag] %s worst barrier h %+.4f m, worst exact clearance %+.4f m\n",
                        result.unsafeStates > 0 ? "UNSAFE" : "clean  ",
                        worstBarrierValue(barrier, result.motion), result.minClearance);
        return result;
    }

    void writeCsvRow(std::ofstream &out, unsigned long seed, const Problem &problem,
                     const char *method, bool eligible, const Result &result)
    {
        if (!out.is_open())
            return;
        out << seed << ',' << problem.scene << ',' << problem.index << ',' << method << ','
            << (eligible ? 1 : 0) << ',' << (result.solved ? 1 : 0) << ',' << result.seconds << ','
            << result.evaluations << ',' << result.vertices << ',' << result.pathLength << ','
            << result.waypoints << ',' << result.auditedStates << ',' << result.unsafeStates << ','
            << result.minClearance << ',' << result.minSelfOverlap << ',' << result.selfColliding << ','
            << result.misses << ',' << result.radPerCall << ',' << result.coarse << ','
            << result.primarySamples << ',' << result.productiveSamples << ",0,0,0,0\n";
    }
}  // namespace

#ifdef OMPL_MBM_HAVE_VAMP
namespace
{
    using VampRobot = ::vamp::robots::Baxter;
    using VampEnvironment =
        ::vamp::collision::Environment<::vamp::FloatVector<::vamp::FloatVectorWidth>>;

    /// \p margin grows every obstacle, which is how this row is held to the same
    /// standard as the others.
    ///
    /// VAMP reads the original primitives rather than the SDF and has no margin of its
    /// own, so without this it plans to *touching* while `auditMotion()` scores it
    /// against `exactClearance - margin`. That is not a safety difference, it is a
    /// difference in what each row was asked for: measured at a 4 mm margin, VAMP's worst
    /// true clearance over a seed was +3 um -- it never actually reached an obstacle --
    /// yet 46 of its 47 returned paths were counted unsafe. Inflating the obstacles here
    /// asks VAMP for the clearance the audit checks, so the unsafe column means the same
    /// thing in every row.
    ///
    /// Growing a capsule's radius is exactly its Minkowski sum with a ball, so that case
    /// is tight. Growing a cuboid's half-extents squares off what should be rounded
    /// corners, which over-covers by at most `margin * (sqrt(3) - 1)` at a vertex; erring
    /// toward over-covering is the right direction for a safety margin.
    VampEnvironment makeVampEnvironment(const Problem &problem, double margin)
    {
        ::vamp::collision::Environment<float> scalar;
        const auto grow = static_cast<float>(std::max(margin, 0.0));
        for (const Obstacle &obstacle : problem.obstacles)
        {
            const Eigen::Vector3f center = obstacle.position.cast<float>();
            const Eigen::Matrix3f rotation = obstacle.rotation.cast<float>();
            if (obstacle.kind == Obstacle::Kind::Box)
            {
                // Built from the quaternion-derived axes directly: VAMP's eigen_rot helper
                // round-trips a matrix through Eigen's intrinsic Euler decomposition, which
                // changes general MotionBenchMaker orientations.
                const Eigen::Vector3f h =
                    obstacle.halfExtents.cast<float>() + Eigen::Vector3f::Constant(grow);
                scalar.cuboids.emplace_back(center.x(), center.y(), center.z(),
                                            rotation(0, 0), rotation(1, 0), rotation(2, 0),
                                            rotation(0, 1), rotation(1, 1), rotation(2, 1),
                                            rotation(0, 2), rotation(1, 2), rotation(2, 2),
                                            h.x(), h.y(), h.z());
            }
            else
            {
                // A capsule conservatively covers the finite cylinder including its end
                // caps, which is the representation VAMP's own MBM demo uses. The UR5
                // harness special-cases its `box` scene into a cuboid; Baxter's three
                // scenes are all bookshelves, so there is no such case here.
                const Eigen::Vector3f halfAxis =
                    rotation.col(2) * static_cast<float>(obstacle.halfLength);
                scalar.capsules.emplace_back(
                    ::vamp::collision::factory::capsule::endpoints::eigen(
                        center - halfAxis, center + halfAxis,
                        static_cast<float>(obstacle.radius) + grow));
            }
        }
        // VAMP's broad phase stops once an obstacle's minimum distance exceeds the sphere's
        // maximum extent, so the shape arrays must be distance-sorted. The Python add_*
        // wrappers do this; direct emplace_back does not.
        scalar.sort();
        return VampEnvironment(scalar);
    }

    class CountingVampMotionValidator : public ob::MotionValidator
    {
    public:
        CountingVampMotionValidator(const ob::SpaceInformationPtr &si, const VampEnvironment &environment,
                                    std::size_t &samples)
          : ob::MotionValidator(si), environment_(environment), samples_(samples)
        {
        }

        bool checkMotion(const ob::State *start, const ob::State *goal) const override
        {
            const auto a = ompl::vamp::ompl_to_vamp<VampRobot>(start);
            const auto b = ompl::vamp::ompl_to_vamp<VampRobot>(goal);
            const double distance = static_cast<double>((b - a).l2_norm());
            const std::size_t batches = std::max<std::size_t>(
                static_cast<std::size_t>(std::ceil(
                    distance / static_cast<double>(::vamp::FloatVectorWidth) * VampRobot::resolution)),
                1);
            samples_ += batches * ::vamp::FloatVectorWidth;
            return ::vamp::planning::validate_motion<VampRobot, ::vamp::FloatVectorWidth,
                                                     VampRobot::resolution>(a, b, environment_);
        }

        bool checkMotion(const ob::State *start, const ob::State *goal,
                         std::pair<ob::State *, double> &lastValid) const override
        {
            lastValid.first = nullptr;
            lastValid.second = 0.0;
            return checkMotion(start, goal);
        }

    private:
        const VampEnvironment &environment_;
        std::size_t &samples_;
    };

    /// VAMP's native SIMD state and motion validation. It reads the original primitives
    /// rather than the SDF, so it is the only row here whose geometry is not the baked
    /// grid -- but it is held to the same margin as every other row; see
    /// `makeVampEnvironment()` for why that is not optional.
    Result runVamp(const Robot &robot, const std::vector<Obstacle> &obstacles, const Problem &problem,
                   double margin, double range, double timeLimit, std::uint_fast32_t seed)
    {
        const VampEnvironment environment = makeVampEnvironment(problem, margin);
        if (std::getenv("OMPL_VAMP_DIAG") != nullptr)
        {
            const auto check = [&environment](const Configuration &q)
            {
                alignas(VampRobot::Configuration::S::Alignment)
                    std::array<float, VampRobot::dimension> buffer{};
                for (std::size_t j = 0; j < VampRobot::dimension; ++j)
                    buffer[j] = static_cast<float>(q[static_cast<Eigen::Index>(j)]);
                const typename VampRobot::Configuration c(buffer.data());
                return ::vamp::planning::validate_motion<VampRobot, ::vamp::FloatVectorWidth, 1>(
                    c, c, environment);
            };
            std::printf("    [vamp diag] %s#%d cuboids+capsules built, start %s, goal %s\n",
                        problem.scene.c_str(), problem.index,
                        check(problem.start) ? "valid" : "INVALID",
                        check(problem.goal) ? "valid" : "INVALID");
        }
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *s) { return std::make_shared<SeededSampler>(s, seed + 1); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        std::size_t samples = 0;
        si->setStateValidityChecker(
            [&environment](const ob::State *state)
            {
                const auto q = ompl::vamp::ompl_to_vamp<VampRobot>(state);
                return ::vamp::planning::validate_motion<VampRobot, ::vamp::FloatVectorWidth, 1>(
                    q, q, environment);
            });
        si->setMotionValidator(
            std::make_shared<CountingVampMotionValidator>(si, environment, samples));
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> startState(space), goalState(space);
        for (int j = 0; j < dimension; ++j)
        {
            startState[j] = problem.start[j];
            goalState[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(startState, goalState, 0.05);
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(range);
        planner->setProblemDefinition(pdef);
        planner->setup();

        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        Result result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.evaluations = samples;
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();
        result.solved = status == ob::PlannerStatus::EXACT_SOLUTION && pdef->hasSolution();
        if (!result.solved)
            return result;

        auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        solution->interpolate(static_cast<unsigned int>(
            std::max(2.0, solution->length() / auditResolution)));
        result.waypoints = solution->getStateCount();
        for (std::size_t i = 0; i < solution->getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(solution->getState(i)));
        result.pathLength = pathLength(result.motion);
        auditMotion(result, robot, obstacles, margin);
        return result;
    }
}  // namespace
#endif

namespace
{
    struct Tally
    {
        std::array<std::size_t, comparisonRows> solved{};
        std::array<std::size_t, comparisonRows> eligible{};
        std::array<std::vector<double>, comparisonRows> milliseconds;
        std::array<double, comparisonRows> travel{};
        std::array<std::size_t, comparisonRows> evaluations{};
        std::array<std::size_t, comparisonRows> vertices{};
        std::array<double, comparisonRows> radPerCall{};
        std::array<double, comparisonRows> coarse{};
        std::array<double, comparisonRows> worstClearance;
        std::array<double, comparisonRows> worstSelf;
        std::array<std::size_t, comparisonRows> unsafe{};
        std::array<std::size_t, comparisonRows> audited{};
        std::array<std::size_t, comparisonRows> selfColliding{};
        std::array<std::size_t, comparisonRows> misses{};

        Tally()
        {
            worstClearance.fill(std::numeric_limits<double>::infinity());
            worstSelf.fill(std::numeric_limits<double>::infinity());
        }

        void add(int row, const Result &r)
        {
            ++eligible[row];
            if (r.solved)
            {
                ++solved[row];
                milliseconds[row].push_back(r.seconds * 1e3);
                travel[row] += r.pathLength;
            }
            evaluations[row] += r.evaluations;
            vertices[row] += r.vertices;
            radPerCall[row] += r.radPerCall;
            coarse[row] += r.coarse;
            worstClearance[row] = std::min(worstClearance[row], r.minClearance);
            worstSelf[row] = std::min(worstSelf[row], r.minSelfOverlap);
            unsafe[row] += r.unsafeStates;
            audited[row] += r.auditedStates;
            selfColliding[row] += r.selfColliding;
            misses[row] += r.misses;
        }
    };

    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty())
            return std::numeric_limits<double>::quiet_NaN();
        std::sort(values.begin(), values.end());
        return values[static_cast<std::size_t>(fraction * (values.size() - 1))];
    }

    void reportRow(const char *label, const Tally &tally, int row)
    {
        const std::size_t n = std::max<std::size_t>(tally.eligible[row], 1);
        std::printf("  %-11s %3zu/%-4zu %8.2f/%8.2f/%9.2f %9zu %6zu %7.2f %8.4f %5.0f%% %10.4f "
                    "%7zu/%-8zu %6zu %9.4f %6zu\n",
                    label, tally.solved[row], tally.eligible[row],
                    percentile(tally.milliseconds[row], 0.0),
                    percentile(tally.milliseconds[row], 0.5),
                    percentile(tally.milliseconds[row], 1.0),
                    tally.evaluations[row] / n, tally.vertices[row] / n,
                    tally.solved[row] > 0 ? tally.travel[row] / tally.solved[row] : 0.0,
                    tally.radPerCall[row] / n, 100.0 * tally.coarse[row] / n,
                    tally.worstClearance[row], tally.unsafe[row], tally.audited[row],
                    tally.misses[row], tally.worstSelf[row], tally.selfColliding[row]);
    }

    void reportHeader()
    {
        std::printf("  planner      solved   ms(min/med/max)          samples vertices    path "
                    "rad/call coarse  worst clr unsafe/audited missed worst self collide\n");
    }
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s scenes.txt [perScene] [seconds] [voxel] [stepSize] [range]\n"
                    "       [margin] [buffer] [seed] [csvPath] [baselineResolution]\n"
                    "       [maxStepScale] [safeHops]\n\n"
                    "Generate scenes.txt with:\n"
                    "  python3 scripts/mbm_to_scenes.py "
                    "external/vamp/resources/baxter/problems.json scenes.txt baxter\n",
                    argv[0]);
        return 1;
    }

    const std::string path = argv[1];
    const int perScene = argc > 2 ? std::atoi(argv[2]) : 20;
    const double timeLimit = argc > 3 ? std::atof(argv[3]) : 5.0;
    const double voxel = argc > 4 ? std::atof(argv[4]) : 0.03;
    const double stepSize = argc > 5 ? std::atof(argv[5]) : 0.05;
    // VAMP publishes a per-robot RRT range and Baxter's is 0.5, against the UR5's 1.5
    // (external/vamp/src/vamp/constants.py). This is not a tuning preference: at the UR5's
    // range every row here solves nothing at all, because a 2 rad straight-line extension
    // in a 14-joint space reaching into a bookshelf is rejected essentially always. VAMP's
    // own evaluation solves 466 of 495 of these in 3.4 s at 0.5, and none of them at 2.0.
    const double range = argc > 6 ? std::atof(argv[6]) : 0.5;
    // The margin the motion is *audited* at, and the extra the barrier guards on top.
    // The filter has replaced the collision checker here, so it must over-reserve against
    // the field's interpolation error: a bookshelf board is 20 mm thick and a voxel is
    // 30 mm, so a barrier held at the audited margin alone is not a barrier at all.
    // A negative buffer asks for one voxel, which is what bounds that error.
    const double margin = argc > 7 ? std::atof(argv[7]) : Barrier::defaultWorldMargin;
    const double buffer = argc > 8 ? std::atof(argv[8]) : -1.0;
    const unsigned long seed = argc > 9 ? std::strtoul(argv[9], nullptr, 10) : 1UL;
    const std::string csvPath = argc > 10 ? argv[10] : std::string();
    // OMPL's default (0.01 of the extent) is far coarser than the audit on a 14-joint
    // space, which would score the baseline unsafe for a reason that has nothing to do
    // with the comparison. Both rows have to be audit-clean for the times to mean anything.
    const double baselineResolution = argc > 11 ? std::atof(argv[11]) : 0.001;
    // How many `stepSize`s a certified hop may cover. Negative asks for no cap, which is
    // the state space's own default; 1 disables hopping altogether and is what the
    // `cbf-nocert` row uses regardless of what is passed here.
    const double maxStepScale = argc > 12 && std::atof(argv[12]) >= 1.0
                                    ? std::atof(argv[12])
                                    : std::numeric_limits<double>::infinity();
    const bool safeHops = argc > 13 ? std::atoi(argv[13]) != 0 : true;

    const std::vector<Problem> problems = readProblems(path);
    const Robot robot(Eigen::Isometry3d::Identity());

    std::printf("\nMotionBenchMaker Baxter, %zu problems loaded, up to %d per scene\n",
                problems.size(), perScene);
    std::printf("voxel %.3f m, stepSize %.3f s, range %.2f rad, %.1f s limit, seed %lu\n",
                voxel, stepSize, range, timeLimit, seed);
    std::printf("audited margin %.1f mm + %s buffer, self margin %.1f mm, kappa %.0f /s\n",
                margin * 1e3, buffer < 0.0 ? "one-voxel" : "given", Barrier::defaultSelfMargin * 1e3,
                Filter::kappa);
    std::printf("%zu spheres, %zu self-pairs in the model\n", Robot::nSpheres, Robot::nSelfPairs);
    std::printf("hop certificate: %s, maxStepScale %s\n\n", safeHops ? "safe (region)" : "no-op",
                std::isinf(maxStepScale) ? "uncapped" : std::to_string(maxStepScale).c_str());

    std::ofstream csvOut;
    if (!csvPath.empty())
    {
        csvOut.open(csvPath);
        csvOut << "seed,scene,problem,method,eligible,solved,seconds,samples,vertices,path_length,"
                  "waypoints,audited_states,unsafe_states,min_clearance,min_self_overlap,"
                  "self_colliding,misses,rad_per_call,coarse_fraction,primary_samples,"
                  "productive_samples,vertices_per_sample,picard_attempts,picard_accepted,"
                  "picard_fallbacks\n";
    }

    std::map<std::string, int> taken;
    Tally overall;
    std::map<std::string, Tally> perSceneTally;
    std::size_t skippedClearance = 0;
    std::vector<double> endpointClearance;

    for (const Problem &problem : problems)
    {
        if (taken[problem.scene] >= perScene)
            continue;
        ++taken[problem.scene];

        // Eligibility first: it reads the original primitives, not the grid, so there is
        // no reason to bake a dense field for a problem that is about to be skipped. At a
        // 7.5 mm voxel that field is 355 MB and several seconds.
        const double guard = buffer < 0.0 ? voxel : buffer;
        const double startClear = exactClearance(robot, problem.obstacles, problem.start);
        const double goalClear = exactClearance(robot, problem.obstacles, problem.goal);
        endpointClearance.push_back(std::min(startClear, goalClear));
        if (startClear < margin + guard || goalClear < margin + guard)
        {
            ++skippedClearance;
            Result skipped;
            writeCsvRow(csvOut, seed, problem, "isSafe", false, skipped);
            writeCsvRow(csvOut, seed, problem, "cbfRRTC", false, skipped);
            writeCsvRow(csvOut, seed, problem, "cbfNoCert", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpFreeGate", false, skipped);
#ifdef OMPL_MBM_HAVE_VAMP
            writeCsvRow(csvOut, seed, problem, "VAMP", false, skipped);
#endif
            continue;
        }

        const sdf::GridSDF field(problem.field(), reachableBounds(), voxel);
        // The barrier the filter enforces is buffered; the audit below is not. That is the
        // whole point of the buffer -- auditing the returned motion against the unbuffered
        // margin is what says the over-reservation was enough.
        const Barrier barrier(robot, field, problem.start, margin + guard,
                              Barrier::defaultSelfMargin);

        if (std::getenv("OMPL_ENDPOINT_DIAG") != nullptr)
        {
            typename Barrier::Values threshold = Barrier::Values::Zero();
            typename Barrier::Evaluation ev;
            const auto worst = [&](const Configuration &q)
            {
                barrier.evaluateScreened(q, threshold, ev);
                return ev.values.head(static_cast<Eigen::Index>(barrier.constraintCount())).minCoeff();
            };
            std::printf("    [endpoints] %s#%d  barrier h: start %+.4f  goal %+.4f\n",
                        problem.scene.c_str(), problem.index, worst(problem.start), worst(problem.goal));
        }
        const auto problemSeed = static_cast<std::uint_fast32_t>(seed * 1000003UL +
                                                                 static_cast<unsigned>(problem.index));
        const Result checked = runChecked(robot, field, problem.obstacles, problem, margin,
                                          Barrier::defaultSelfMargin, range, timeLimit,
                                          baselineResolution, problemSeed);
        // One filter object per row, so the call counters and the QP's cached state are
        // never shared between two measurements.
        const Filter qpFilter(barrier);
        const Filter qpFilterNoCert(barrier);
        const ompl::demo::RobotQPFreeGate<Robot> gate(barrier);
        const Result filtered = runFiltered(robot, barrier, qpFilter, problem.obstacles, problem,
                                            margin, stepSize, range, timeLimit, problemSeed,
                                            maxStepScale, safeHops);
        // Same QP, same everything, hop certificate disabled.
        const Result noCert = runFiltered(robot, barrier, qpFilterNoCert, problem.obstacles, problem,
                                          margin, stepSize, range, timeLimit, problemSeed, 1.0, safeHops);
        // The gate issues no certificate, so its hop length is `stepSize` regardless; the
        // cap is set to 1 to say so rather than to rely on that.
        const Result gated = runFiltered(robot, barrier, gate, problem.obstacles, problem, margin,
                                         stepSize, range, timeLimit, problemSeed, 1.0, safeHops);

        Tally &scene = perSceneTally[problem.scene];
        scene.add(checkedRow, checked);
        scene.add(cbfRow, filtered);
        scene.add(noCertRow, noCert);
        scene.add(gateRow, gated);
        overall.add(checkedRow, checked);
        overall.add(cbfRow, filtered);
        overall.add(noCertRow, noCert);
        overall.add(gateRow, gated);
        writeCsvRow(csvOut, seed, problem, "isSafe", true, checked);
        writeCsvRow(csvOut, seed, problem, "cbfRRTC", true, filtered);
        writeCsvRow(csvOut, seed, problem, "cbfNoCert", true, noCert);
        writeCsvRow(csvOut, seed, problem, "qpFreeGate", true, gated);
#ifdef OMPL_MBM_HAVE_VAMP
        const Result vamp = runVamp(robot, problem.obstacles, problem, margin, range, timeLimit,
                                    problemSeed);
        scene.add(vampRow, vamp);
        overall.add(vampRow, vamp);
        writeCsvRow(csvOut, seed, problem, "VAMP", true, vamp);
#endif
    }

    for (const auto &[scene, tally] : perSceneTally)
    {
        std::printf("%s  (%zu problems)\n", scene.c_str(), tally.eligible[checkedRow]);
        reportHeader();
        reportRow("rrtconnect", tally, checkedRow);
        reportRow("cbf-rrtc", tally, cbfRow);
        reportRow("cbf-nocert", tally, noCertRow);
        reportRow("qp-free", tally, gateRow);
#ifdef OMPL_MBM_HAVE_VAMP
        reportRow("vamp-rrtc", tally, vampRow);
#endif
        std::printf("\n");
    }
    std::printf("all scenes  (%zu problems, %zu skipped: endpoint inside the guarded margin)\n",
                overall.eligible[checkedRow], skippedClearance);
    if (!endpointClearance.empty())
    {
        std::vector<double> sorted = endpointClearance;
        std::sort(sorted.begin(), sorted.end());
        const auto share = [&](double m)
        {
            return 100.0 * static_cast<double>(
                       std::lower_bound(sorted.begin(), sorted.end(), m) - sorted.begin()) /
                   static_cast<double>(sorted.size());
        };
        std::printf("endpoint clearance over %zu problems: min %.1f, p50 %.1f, max %.1f mm; "
                    "ineligible at a guarded margin of 5/10/20/34 mm: %.0f%%/%.0f%%/%.0f%%/%.0f%%\n",
                    sorted.size(), sorted.front() * 1e3, sorted[sorted.size() / 2] * 1e3,
                    sorted.back() * 1e3, share(0.005), share(0.010), share(0.020), share(0.034));
    }
    reportHeader();
    reportRow("rrtconnect", overall, checkedRow);
    reportRow("cbf-rrtc", overall, cbfRow);
    reportRow("cbf-nocert", overall, noCertRow);
    reportRow("qp-free", overall, gateRow);
#ifdef OMPL_MBM_HAVE_VAMP
    reportRow("vamp-rrtc", overall, vampRow);
#endif
    std::printf("\n\"unsafe/audited\" evaluates exact primitive clearance at every state of the\n"
                "densified solution -- for the CBF row that is the rollout the planner recorded,\n"
                "replayed. Non-zero unsafe invalidates a row however fast it was.\n");
    return 0;
}
