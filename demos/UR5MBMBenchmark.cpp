// demos/UR5MBMBenchmark.cpp
//
// The CBF rollout against ordinary collision-checked planning and VAMP on a *standard* problem
// set: MotionBenchMaker's UR5 scenes, 689 problems over 7 scenes, as shipped by VAMP.
//
//     ./scripts/mbm_to_scenes.py /path/to/vamp/resources/ur5/problems.json scenes.txt
//     ./build/demos/demo_UR5MBMBenchmark scenes.txt [perScene] [seconds] [voxel] [stepSize]
//         [range] [margin] [buffer] [segmentFraction] [kappa] [maxStepScale] [selfMargin]
//         [pathPrefix] [shortcutDelta] [seed] [csvPath] [safeHops]
//         [picardIterations] [picardWindow] [picardWorkers] [trajectoryPrefixes]
//         [rolloutCallBudget]
//
// `pathPrefix` dumps the audited motions (`<prefix>.rrtc`, `<prefix>.cbf`, `<prefix>.vamp`) so
// they can be replayed against the real UR5 meshes in PyBullet:
//
//     ur5_experiments/scripts/audit_self_collision.py --path <prefix>.cbf
//
// That is the only check that does not share the sphere model with the `collide` column.
//
// The last two are the certified-step A/B: `maxStepScale 1` pins the rollout to a fixed
// `stepSize` however much room it has, and `kappa` is the CBF decay rate in 1/s, which
// sets how fast clearance may be spent and so scales the certificate with it. It replaces
// the old per-step `gamma`; the default 8 /s is what `gamma 0.4` amounted to at the
// default 0.05 s step, so a default run is unchanged.
//
// Why this and not UR5CBFPlanningDemo's scene: that one is a pair of spheres placed to
// block the direct sweep, and `geometric::RRTConnect` solves it in six vertices. A
// method whose claim is "no wasted edges" cannot show anything on a problem with no
// wasted edges. These scenes are cluttered, externally defined, and widely reported, so
// they can neither be tuned to flatter the filter nor dismissed as a strawman.
//
// Every obstacle in the set is a box or a cylinder, both of which have an exact
// closed-form signed distance -- so the field the barrier reads is exact up to the grid,
// with no mesh, no FCL, and no sign ambiguity.
//
// ### What is compared
//
// - `rrtconnect`: stock geometric RRTConnect, straight-line edges, the SDF behind an
//   ordinary StateValidityChecker. The bar.
// - `cbf-rrtc`: the same planner over `cbf::FilteredStateSpace`, so every edge is a CBF
//   rollout and every intermediate state is certified as it is produced. No state
//   validity checker at all.
// - `qp-fixed`: the same CBF-QP and rollout, capped at one integration step so no
//   Lipschitz certificate is spent to skip later filter calls.
// - `qp-free`: the accept-or-stop CBF gate adapted from LQR-CBF-RRT*. It applies the
//   nominal control only while every potentially binding CBF row passes, never solves a
//   QP, and has no state validity checker.
// - `vamp-rrtc`: the same planner with VAMP's native SIMD state and motion validation,
//   when this target is built with `OMPL_BUILD_VAMP=ON`.
//
// All see the same original primitives, sphere robot, start and goal. The SDF methods
// use the requested margin; native VAMP is a zero-margin comparison.
//
// ### Feasibility is reported, not assumed
//
// MotionBenchMaker calls a problem valid when the *mesh* robot is collision free. Our
// robot is 40 spheres that do not enclose those meshes (see ClearanceBarrier) and the
// barrier adds a margin on top, so a problem can be valid upstream and still have its
// start or goal inside our margin. Those are counted and excluded rather than scored as
// failures -- and the count is itself a result, because a margin that rules out most of
// a standard benchmark is a finding about the margin.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/CertifiedRegionRollout.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/ParallelPicardRollout.h>
#include <ompl/cbf/Profiler.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>
#include <ompl/util/Time.h>

#include "UR5SelfCollisionAudit.h"
#include "UR5QPFreeGate.h"
#include "EnvelopeHoldFilter.h"
#include "HoldTimeRegionRollout.h"

#ifdef OMPL_MBM_HAVE_VAMP
#include <ompl/vamp/Utils.h>
#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/ur5.hh>
#endif

namespace ob = ompl::base;
namespace og = ompl::geometric;
using Barrier = ompl::cbf::ClearanceBarrier;
using Filter = ompl::cbf::CBFControlFilter;
using Space = ompl::cbf::FilteredStateSpace;
using UR5 = ompl::robots::UR5;
namespace sdf = ompl::sdf;

namespace
{
    constexpr int dimension = 6;
    constexpr int checkedRow = 0;
    constexpr int qpFixedRow = 1;
    constexpr int qpAdaptiveRow = 2;
    constexpr int qpLipschitzRow = 3;
    constexpr int qpSafeRow = 4;
    constexpr int qpFreeRow = 5;
    constexpr int qpEnvelopeRow = 6;
    constexpr int vampRow = 7;
    /// The unassisted QP: `qpFixed`'s controller with every QP-side optimization off.
    constexpr int qpPlainRow = 8;
    constexpr int comparisonRows = 9;

    /// Joint-space spacing all rows are audited at, in radians. Finer than the rollout
    /// step so the audit is not merely re-reading the filter's own decisions.
    /// The goal region's radius, in radians. The baseline and VAMP rows have always used
    /// 0.05 and the filtered rows 0.1 -- a filtered rollout cannot land exactly on a
    /// target, so it was given slack. That slack is an asymmetry in the CBF rows' favour
    /// on both solve time and path length, so it is overridable: set
    /// `OMPL_MBM_GOAL_TOLERANCE` to charge every row the same radius.
    double goalTolerance(double fallback)
    {
        static const double override_ = []
        {
            if (const char *v = std::getenv("OMPL_MBM_GOAL_TOLERANCE"))
            {
                const double parsed = std::atof(v);
                if (parsed > 0.0)
                    return parsed;
            }
            return -1.0;
        }();
        return override_ > 0.0 ? override_ : fallback;
    }

    /// Overridable because the audit's own spacing bounds what the audit can say. At
    /// 0.02 rad a sample is up to ~2 cm of workspace motion at a 1 m lever arm -- an
    /// order of magnitude more than the millimetre-scale margins under test -- so
    /// "zero unsafe" means "no sampled state penetrated", not "the path never did".
    /// `OMPL_MBM_AUDIT_RES=0.002` puts it under a 3 mm buffer, at roughly ten times the
    /// audit cost, which is what decides whether a clean column is real or an artefact
    /// of looking too coarsely.
    double auditResolutionValue()
    {
        static const double value = []
        {
            if (const char *v = std::getenv("OMPL_MBM_AUDIT_RES"))
            {
                const double parsed = std::atof(v);
                if (parsed > 0.0)
                    return parsed;
            }
            return 0.02;
        }();
        return value;
    }

    /// One obstacle: a box (`halfExtents`) or a cylinder (`radius`, `halfLength` about
    /// the local z axis), posed in the world.
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

        /// Exact signed distance. Both formulas are the standard ones: reduce to the
        /// box's local frame, take the per-axis overshoot, then the norm of its positive
        /// part outside and the largest (negative) component inside.
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
        UR5::Configuration start{UR5::Configuration::Zero()};
        UR5::Configuration goal{UR5::Configuration::Zero()};
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

    /// The per-joint speed limit the filter, the state space and the baseline's edge
    /// resolution all read. `UR5::velocityLimits()` is a conservative `Constant(0.5)`,
    /// well under the real arm (2.09 rad/s proximal, 3.14 wrist); `OMPL_CBF_MAX_SPEED`
    /// replaces it with a uniform value so a run can ask what the cap costs.
    ///
    /// It cannot be zero or unbounded. `FilteredStateSpace` rejects non-positive entries,
    /// and the QP's control box is what makes its feasible set bounded. Raising it is
    /// therefore how "no velocity limit" is expressed, and it moves three things at once:
    /// the QP's box and `decreaseRates`, and so every certificate length. It does *not*
    /// change what the planner's range means: `Operations::distance` takes a speed
    /// argument and discards it, returning radians, so `range` is radians for every row;
    /// `Operations::duration` is the seconds-valued one and only sets a rollout's
    /// horizon. It does change `defaultReachTolerance`, which is `maxSpeed.norm() *
    /// stepSize` -- at 10 rad/s and a 10 ms step that is 0.245 rad of slack on "did the
    /// rollout arrive", and the audit failure it caused (worst clearance -34.8 mm) is why
    /// a run that raises this should set the tolerance explicitly.
    ///
    /// Pass an explicit `segmentFraction` alongside it, or the baseline's collision-check
    /// spacing is derived from this and silently coarsens with it.
    const UR5::Configuration &effectiveMaxSpeed()
    {
        static const UR5::Configuration value = []
        {
            if (const char *v = std::getenv("OMPL_CBF_MAX_SPEED"))
            {
                const double speed = std::atof(v);
                if (speed > 0.0)
                    return UR5::Configuration::Constant(speed).eval();
            }
            return UR5::velocityLimits().eval();
        }();
        return value;
    }

    /// Which comparison rows to run, so a study of a subset does not pay for the rest.
    /// `OMPL_MBM_ROWS=isSafe,qpAdaptive,qpEnvelope`; unset runs everything, which is the
    /// behaviour every earlier result was produced with.
    ///
    /// The selection governs the summary table and the CSV as well as the work: a
    /// deselected row is absent from both, rather than present as a line of zeros that
    /// reads as a row which ran and solved nothing. Names are the CSV's own `method`
    /// values -- `isSafe`, `qpFixed`, `qpAdaptive`, `l1Old`, `holdNew`, `qpFreeGate`,
    /// `qpEnvelope`, `VAMP`, `qpPlain`.
    bool wantRow(const char *name)
    {
        static const std::string selection = []
        {
            const char *v = std::getenv("OMPL_MBM_ROWS");
            return v != nullptr ? std::string(v) : std::string();
        }();
        if (selection.empty())
            return true;
        const std::string needle(name);
        std::size_t at = selection.find(needle);
        while (at != std::string::npos)
        {
            const bool leftOk = at == 0 || selection[at - 1] == ',';
            const std::size_t end = at + needle.size();
            const bool rightOk = end == selection.size() || selection[end] == ',';
            if (leftOk && rightOk)
                return true;
            at = selection.find(needle, at + 1);
        }
        return false;
    }

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
                UR5::Configuration q;
                for (int j = 0; j < dimension; ++j)
                    fields >> q[j];
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

    ob::RealVectorBounds jointBounds()
    {
        ob::RealVectorBounds bounds(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, UR5::lowerBounds()[j]);
            bounds.setHigh(j, UR5::upperBounds()[j]);
        }
        return bounds;
    }

    class SeededUR5Sampler final : public ob::RealVectorStateSampler
    {
    public:
        SeededUR5Sampler(const ob::StateSpace *space, std::uint_fast32_t seed)
          : ob::RealVectorStateSampler(space)
        {
            rng_.setLocalSeed(seed);
        }
    };

    struct Result
    {
        bool solved{false};
        double seconds{0.0};
        std::size_t evaluations{0};  ///< sampled configurations, barrier calls, or SIMD lanes
        std::size_t vertices{0};
        std::size_t primarySamples{0};
        std::size_t productiveSamples{0};
        std::size_t picardAttempts{0};
        std::size_t picardAccepted{0};
        std::size_t picardFallbacks{0};
        double pathLength{0.0};
        std::size_t waypoints{0};
        std::size_t unsafeStates{0};
        std::size_t auditedStates{0};
        std::size_t misses{0};  ///< solution edges re-derived rather than replayed
        /// QP work this row actually did, as deltas over its own planning call -- see
        /// ControlFilter::Counters. `qpRows` is summed over calls, so `qpRows / qpCalls`
        /// is the program size the filter assembled per call, which is the quantity
        /// screening and the active-set pair traversal are supposed to move; and
        /// `qpSolves / qpCalls` is the share of calls that got past the feasibility
        /// bypass and the closed-form projection to reach qpmad. A row that never builds
        /// a QP -- `isSafe`, `qpFreeGate` -- leaves all three at zero, which is the
        /// honest reading of "no QP" rather than a missing measurement.
        std::size_t qpCalls{0};
        std::size_t qpRows{0};
        std::size_t qpSolves{0};
        /// Joint-space radians per filter call, and the share of calls that ran past
        /// stepSize on a certificate. Meaningless for the baseline, whose "evaluation"
        /// is a collision check at a fixed resolution rather than a step.
        double radPerCall{0.0};
        double coarse{0.0};
        /// Share of hops whose length was set by, respectively: the one-step floor (the
        /// certified region was shorter than a step), the region itself, and the end of
        /// the extension. They sum to one over rows that roll. This is what says whether
        /// a longer certificate can help at all -- only the first bucket can convert.
        double hopFloored{0.0};
        double hopAtRegion{0.0};
        double hopEdgeLimited{0.0};
        double minClearance{std::numeric_limits<double>::infinity()};
        /// The independent self-collision check -- see demos/UR5SelfCollisionAudit.h. The
        /// barrier carries self rows now, but only for the pairs the offline search kept,
        /// so this walks all of them and is the only thing able to say that search was
        /// wrong.
        double minSelfOverlap{std::numeric_limits<double>::infinity()};
        std::size_t selfColliding{0};
    };

    void recordPathShape(const og::PathGeometric &path, Result &result)
    {
        result.pathLength = path.length();
        result.waypoints = path.getStateCount();
    }

    double exactWorldClearance(const Problem &problem, const UR5 &robot,
                               const UR5::Configuration &q, double margin)
    {
        const UR5::Kinematics kin = robot.kinematics(q);
        double worst = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < UR5::nSpheres; ++i)
        {
            const Eigen::Vector3d center = UR5::sphereCenter(kin, i);
            for (const Obstacle &obstacle : problem.obstacles)
                worst = std::min(worst, obstacle.distance(center) - UR5::spheres()[i].radius - margin);
        }
        return worst;
    }

    void auditStraight(og::PathGeometric path, const Problem &problem, const Barrier &barrier,
                       Result &result,
                       std::vector<UR5::Configuration> *record)
    {
        recordPathShape(path, result);
        const unsigned int segments = static_cast<unsigned int>(
            std::max(1.0, std::ceil(path.length() / auditResolutionValue())));
        path.interpolate(segments + 1);
        result.auditedStates = path.getStateCount();
        for (std::size_t i = 0; i < path.getStateCount(); ++i)
        {
            const UR5::Configuration q = Space::configurationOf(path.getState(i));
            const double h = exactWorldClearance(problem, barrier.robot(), q, barrier.margin());
            result.minClearance = std::min(result.minClearance, h);
            result.unsafeStates += h < 0.0 ? 1 : 0;
            const double own = ompl::demo::worstSelfOverlap(barrier.robot(), q);
            result.minSelfOverlap = std::min(result.minSelfOverlap, own);
            result.selfColliding += own < 0.0 ? 1 : 0;
            if (record != nullptr)
                record->push_back(q);
        }
    }

    /// Audit a solution the way the CBF rows must be audited: expand it into the motion
    /// that would actually be executed, then evaluate exact primitive clearance at every
    /// state of it (not the interpolated planning SDF).
    ///
    /// `record`, when given, collects exactly the states audited here, so an external mesh
    /// checker is handed the same motion at the same resolution that the `collide` column
    /// is computed from. Anything coarser would let a mesh contact hide between the states
    /// the sphere model was scored on.
    void audit(const og::PathGeometric &solution, const Problem &problem, const Barrier &barrier,
               Result &result,
               std::vector<UR5::Configuration> *record)
    {
        // executedPath() replays the rollout the planner recorded for each edge, so this
        // is the executed motion rather than a reconstruction of it -- and it honours the
        // requested resolution. PathGeometric::interpolate() would not: it budgets states
        // by the Euclidean distance() between waypoints, which is a lower bound on the arc
        // length of a deflected edge, so it under-samples exactly where the CBF was
        // working hardest. `misses` must be zero or the replay claim is void.
        //
        // The resolution is the one the baseline is audited at, so the clearance columns
        // mean the same thing in all rows. Note this genuinely samples *inside* a step
        // now; the old rollout quantised every fraction to a step boundary, so a fine
        // request silently came back at the step size.
        const og::PathGeometric path = ompl::cbf::executedPath(solution, auditResolutionValue(), &result.misses);
        recordPathShape(path, result);
        result.auditedStates = path.getStateCount();
        for (std::size_t i = 0; i < path.getStateCount(); ++i)
        {
            const UR5::Configuration q = Space::configurationOf(path.getState(i));
            const double h = exactWorldClearance(problem, barrier.robot(), q, barrier.margin());
            result.minClearance = std::min(result.minClearance, h);
            if (h < 0.0)
                ++result.unsafeStates;

            const double own = ompl::demo::worstSelfOverlap(barrier.robot(), q);
            result.minSelfOverlap = std::min(result.minSelfOverlap, own);
            if (own < 0.0)
                ++result.selfColliding;

            if (record != nullptr)
                record->push_back(q);
        }
    }

    /// The bar: straight-line edges, the same SDF behind an ordinary validity checker.
    Result runCollisionChecked(const Problem &problem, const Barrier &barrier, double range,
                               double timeLimit, double segmentFraction, double shortcutDelta,
                               std::uint_fast32_t sampleSeed,
                               std::vector<UR5::Configuration> *record)
    {
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [sampleSeed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededUR5Sampler>(stateSpace, sampleSeed);
            });
        if (segmentFraction > 0.0)
            space->setLongestValidSegmentFraction(segmentFraction);
        auto si = std::make_shared<ob::SpaceInformation>(space);

        std::size_t checks = 0;
        si->setStateValidityChecker([&barrier, &checks](const ob::State *state)
                                    {
                                        ++checks;
                                        return barrier.isSafe(Space::configurationOf(state));
                                    });
        si->setup();

        ob::ScopedState<> start(space), goal(space);
        for (int j = 0; j < dimension; ++j)
        {
            start[j] = problem.start[j];
            goal[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(start, goal, goalTolerance(0.05));

        Result result;
        // Re-seed the global seed generator immediately before the planner is built.
        // `RRTConnect` holds its own `RNG`, which draws `nextSeed()` from that generator
        // at construction, so without this a planner's stream depends on how many RNGs
        // the process happened to build before it -- and therefore on which *other* rows
        // `OMPL_MBM_ROWS` selected. That made row-set composition perturb every row:
        // `qpAdaptive` scored 1523 alone and 1484 beside `qpEnvelope`. Re-seeding here
        // makes each row a function of `sampleSeed` alone, so a row means the same thing
        // whichever others are enabled.
        ompl::RNG::setSeed(sampleSeed);
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(range);
        planner->setSampleExtensionCallback(
            [&result](const ob::State *, const ob::State *, bool, bool productive, bool, bool)
            {
                ++result.primarySamples;
                result.productiveSamples += productive ? 1u : 0u;
            });
        planner->setProblemDefinition(pdef);
        planner->setup();

        const ompl::time::point begin = ompl::time::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        result.seconds = ompl::time::seconds(ompl::time::now() - begin);
        result.solved = (status == ob::PlannerStatus::EXACT_SOLUTION);
        result.evaluations = checks;

        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();

        if (result.solved)
        {
            // Audited at the same resolution the rollout uses, so the clearance columns
            // mean the same thing in all rows.
            og::PathGeometric path(*std::static_pointer_cast<og::PathGeometric>(
                pdef->getSolutionPath()));
            if (shortcutDelta > 0.0)
            {
                // Stock RRT-Rope. This row's edges *are* straight lines, so OMPL's own
                // implementation is exactly right for it -- but all rows must be shortcut
                // at the same delta, or "shorter" measures rope's aggressiveness rather
                // than the filter.
                og::PathSimplifier simplifier(si);
                simplifier.ropeShortcutPath(path, shortcutDelta);
            }
            auditStraight(path, problem, barrier, result, record);
        }
        return result;
    }

#ifdef OMPL_MBM_HAVE_VAMP
    using VampRobot = ::vamp::robots::UR5;
    using VampEnvironment =
        ::vamp::collision::Environment<::vamp::FloatVector<::vamp::FloatVectorWidth>>;

    VampEnvironment makeVampEnvironment(const Problem &problem)
    {
        ::vamp::collision::Environment<float> scalar;
        for (const Obstacle &obstacle : problem.obstacles)
        {
            const Eigen::Vector3f center = obstacle.position.cast<float>();
            const Eigen::Matrix3f rotation = obstacle.rotation.cast<float>();
            const auto addCuboid = [&](const Eigen::Vector3f &halfExtents)
            {
                // Construct from the quaternion-derived axes directly.  VAMP's
                // eigen_rot helper converts a matrix through Eigen's intrinsic Euler
                // decomposition and then reconstructs an extrinsic XYZ rotation; that
                // round trip changes general MBM obstacle orientations.
                scalar.cuboids.emplace_back(
                    center.x(), center.y(), center.z(),
                    rotation(0, 0), rotation(1, 0), rotation(2, 0),
                    rotation(0, 1), rotation(1, 1), rotation(2, 1),
                    rotation(0, 2), rotation(1, 2), rotation(2, 2),
                    halfExtents.x(), halfExtents.y(), halfExtents.z());
            };
            if (obstacle.kind == Obstacle::Kind::Box)
                addCuboid(obstacle.halfExtents.cast<float>());
            else if (problem.scene == "box")
            {
                const Eigen::Vector3f halfExtents(static_cast<float>(obstacle.radius),
                                                  static_cast<float>(obstacle.radius),
                                                  static_cast<float>(obstacle.halfLength));
                addCuboid(halfExtents);
            }
            else
            {
                // This is the representation used by VAMP's own MBM demo: a capsule
                // conservatively covers the finite cylinder, including its end caps.
                const Eigen::Vector3f halfAxis =
                    rotation.col(2) * static_cast<float>(obstacle.halfLength);
                scalar.capsules.emplace_back(
                    ::vamp::collision::factory::capsule::endpoints::eigen(
                        center - halfAxis, center + halfAxis, static_cast<float>(obstacle.radius)));
            }
        }
        // VAMP's broad phase stops once an obstacle's minimum distance exceeds the
        // sphere's maximum extent, so the shape arrays must be distance-sorted.  The
        // Python add_* wrappers and VAMP's C++ examples do this automatically; direct
        // emplace_back(), as above, does not.
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

    Result runVamp(const Problem &problem, const Barrier &audited, double range, double timeLimit,
                   double shortcutDelta, std::uint_fast32_t sampleSeed,
                   std::vector<UR5::Configuration> *record)
    {
        const VampEnvironment environment = makeVampEnvironment(problem);
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [sampleSeed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededUR5Sampler>(stateSpace, sampleSeed);
            });
        auto si = std::make_shared<ob::SpaceInformation>(space);

        std::size_t samples = 0;
        si->setStateValidityChecker([&](const ob::State *state)
                                    {
                                        ++samples;
                                        const auto q = ompl::vamp::ompl_to_vamp<VampRobot>(state);
                                        return ::vamp::planning::validate_motion<VampRobot,
                                            ::vamp::FloatVectorWidth, 1>(q, q, environment);
                                    });
        si->setMotionValidator(
            std::make_shared<CountingVampMotionValidator>(si, environment, samples));
        si->setup();

        ob::ScopedState<> start(space), goal(space);
        for (int j = 0; j < dimension; ++j)
        {
            start[j] = problem.start[j];
            goal[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(start, goal, goalTolerance(0.05));
        // Re-seed the global seed generator immediately before the planner is built.
        // `RRTConnect` holds its own `RNG`, which draws `nextSeed()` from that generator
        // at construction, so without this a planner's stream depends on how many RNGs
        // the process happened to build before it -- and therefore on which *other* rows
        // `OMPL_MBM_ROWS` selected. That made row-set composition perturb every row:
        // `qpAdaptive` scored 1523 alone and 1484 beside `qpEnvelope`. Re-seeding here
        // makes each row a function of `sampleSeed` alone, so a row means the same thing
        // whichever others are enabled.
        ompl::RNG::setSeed(sampleSeed);
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(range);
        planner->setProblemDefinition(pdef);
        planner->setup();

        Result result;
        const ompl::time::point begin = ompl::time::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        result.seconds = ompl::time::seconds(ompl::time::now() - begin);
        result.solved = status == ob::PlannerStatus::EXACT_SOLUTION;
        result.evaluations = samples;
        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();

        if (result.solved)
        {
            og::PathGeometric path(
                *std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath()));
            if (shortcutDelta > 0.0)
            {
                og::PathSimplifier simplifier(si);
                simplifier.ropeShortcutPath(path, shortcutDelta);
            }
            auditStraight(path, problem, audited, result, record);
        }
        return result;
    }
#endif

    /// The CBF rollout as the state space's interpolate(), with no collision checking
    /// anywhere: the barrier certifies each step as it is produced.
    Result runFiltered(const Problem &problem, const Barrier &audited,
                       const ompl::cbf::ControlFilter &filter,
                       double stepSize, double range, double timeLimit, double maxStepScale,
                       double shortcutDelta, bool safeHops, unsigned int picardIterations,
                       unsigned int picardWindow, unsigned int picardWorkers,
                       unsigned int trajectoryPrefixes, unsigned int rolloutCallBudget,
                       std::uint_fast32_t sampleSeed,
                       std::vector<UR5::Configuration> *record,
                       const Space::RolloutPlanner *customRollout = nullptr)
    {
        auto space = std::make_shared<Space>(filter, stepSize, effectiveMaxSpeed());
        space->setBounds(jointBounds());
        space->setStateSamplerAllocator(
            [sampleSeed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededUR5Sampler>(stateSpace, sampleSeed);
            });
        if (maxStepScale > 0.0)
            space->setMaxStepScale(maxStepScale);
        // The A/B for the certified region: off spends the no-op certificate, on spends
        // the safety one. Everything else about the row is identical, and the audit
        // column is what says whether the longer hops cost anything.
        space->setSafeHops(safeHops);
        Space::EarlyTermination earlyTermination;
        earlyTermination.enabled = true;
        earlyTermination.maxFilterCalls = rolloutCallBudget;
        space->setEarlyTermination(earlyTermination);

        std::unique_ptr<ompl::cbf::ParallelPicardRollout> picard;
        if (customRollout != nullptr)
            space->setRolloutPlanner(*customRollout);
        else if (picardIterations > 0)
        {
            const auto *qpFilter = dynamic_cast<const Filter *>(&filter);
            if (qpFilter == nullptr)
                throw ompl::Exception("Picard rollout requires the CBF-QP filter");
            ompl::cbf::ParallelPicardRollout::Parameters picardParameters;
            picardParameters.maxIterations = picardIterations;
            picardParameters.windowSteps = picardWindow;
            picardParameters.workers = picardWorkers;
            picard = std::make_unique<ompl::cbf::ParallelPicardRollout>(*qpFilter,
                                                                        picardParameters);
            space->setRolloutPlanner(
                [&picard, space](const UR5::Configuration &from,
                                 const UR5::Configuration &to, double fraction,
                                 Space::Rollout &rollout)
                {
                    return picard->plan(from, to, fraction, space->stepSize(),
                                        space->maxSpeed(), space->maxStepScale(),
                                        space->reachTolerance(), rollout);
                });
        }

        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::FilteredMotionValidator>(si));
        si->setup();

        ob::ScopedState<> start(space), goal(space);
        for (int j = 0; j < dimension; ++j)
        {
            start[j] = problem.start[j];
            goal[j] = problem.goal[j];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        // A rollout cannot be asked to land on an exact state next to an obstacle, so the
        // goal gets a tolerance. Kept small enough that "solved" still means solved.
        pdef->setStartAndGoalStates(start, goal, goalTolerance(0.1));

        Result result;
        // Re-seed the global seed generator immediately before the planner is built.
        // `RRTConnect` holds its own `RNG`, which draws `nextSeed()` from that generator
        // at construction, so without this a planner's stream depends on how many RNGs
        // the process happened to build before it -- and therefore on which *other* rows
        // `OMPL_MBM_ROWS` selected. That made row-set composition perturb every row:
        // `qpAdaptive` scored 1523 alone and 1484 beside `qpEnvelope`. Re-seeding here
        // makes each row a function of `sampleSeed` alone, so a row means the same thing
        // whichever others are enabled.
        ompl::RNG::setSeed(sampleSeed);
        auto planner = std::make_shared<og::RRTConnect>(si, trajectoryPrefixes > 0);
        planner->setRange(range);
        planner->setRetainPartialSteering(true);
        if (trajectoryPrefixes > 0)
        {
            planner->setMaxIntermediateStates(trajectoryPrefixes);
            planner->setIntermediateStatesCallback(
                [space](const std::vector<ob::State *> &states)
                {
                    space->recordSubdivisions(states);
                });
        }
        planner->setSampleExtensionCallback(
            [&result](const ob::State *, const ob::State *, bool, bool productive, bool, bool)
            {
                ++result.primarySamples;
                result.productiveSamples += productive ? 1u : 0u;
            });
        planner->setProblemDefinition(pdef);
        planner->setup();

        // Cumulative on the filter object, and `lipschitzFilter` is handed to three
        // rows, so a row's own QP work is the delta across its solve rather than the
        // running total. Copied, not bound: `counters()` returns a reference into the
        // live filter, which the solve below is about to advance.
        const ompl::cbf::ControlFilter::Counters qpBefore = filter.counters();
        const ompl::time::point begin = ompl::time::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));
        result.seconds = ompl::time::seconds(ompl::time::now() - begin);
        result.solved = (status == ob::PlannerStatus::EXACT_SOLUTION);
        result.evaluations = space->statistics().steps;
        const ompl::cbf::ControlFilter::Counters &qpAfter = filter.counters();
        result.qpCalls = qpAfter.calls - qpBefore.calls;
        result.qpRows = qpAfter.rows - qpBefore.rows;
        result.qpSolves = qpAfter.solves - qpBefore.solves;
        if (picard)
        {
            result.picardAttempts = picard->statistics().attempts;
            result.picardAccepted = picard->statistics().accepted;
            result.picardFallbacks = picard->statistics().fallbacks;
        }
        if (space->statistics().steps > 0)
        {
            const double calls = static_cast<double>(space->statistics().steps);
            result.radPerCall = space->statistics().travel / calls;
            result.coarse = static_cast<double>(space->statistics().coarse) / calls;
            const auto &st = space->statistics();
            const double hops = static_cast<double>(st.hopFloored + st.hopAtRegion +
                                                    st.hopEdgeLimited + st.hopScaleLimited);
            if (hops > 0.0)
            {
                result.hopFloored = static_cast<double>(st.hopFloored) / hops;
                result.hopAtRegion = static_cast<double>(st.hopAtRegion) / hops;
                result.hopEdgeLimited = static_cast<double>(st.hopEdgeLimited) / hops;
            }
        }

        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();

        if (result.solved)
        {
            og::PathGeometric solution(
                *std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath()));
            if (shortcutDelta > 0.0)
                // RRT-Rope with the CBF rollout as the edge test, at the same delta as the
                // baseline's stock rope above.
                ompl::cbf::ropeShortcut(solution, shortcutDelta);
            audit(solution, problem, audited, result, record);
        }
        return result;
    }

    struct Tally
    {
        static constexpr int rows = comparisonRows;
        int attempted{0};
        int skipped{0};  ///< start or goal inside our margin
        /// Breakdown of `skipped` by which barrier rejected the endpoint. Not mutually
        /// exclusive -- a problem can fail both -- so these need not sum to `skipped`.
        /// Cross-checked once against the real UR5 mesh in PyBullet
        /// (ur5_experiments/scripts/audit_self_collision.py): of 117 endpoints the
        /// self-pair barrier flagged on this set, 85 were confirmed mesh self-collisions
        /// (forearm_link overlapping wrist_2_link/wrist_3_link, up to 22 mm) -- a defect
        /// in MotionBenchMaker's own goal poses, not a sphere-model false positive. The
        /// other 32 were not confirmed, i.e. the calibrated per-pair margin is
        /// conservative there. See UR5SelfCollisionAudit.h.
        int skippedClearance{0};
        int skippedSelfCollision{0};
        /// min(start, goal) sphere clearance at zero margin, per problem. VAMP validated
        /// this set against the same 40 spheres with no margin, so these should all be
        /// positive; how far above zero they sit is what decides which margins are
        /// affordable on a standard benchmark.
        std::vector<double> endpointClearance;
        std::array<int, rows> solved{};
        std::array<std::vector<double>, rows> seconds;
        std::array<std::vector<double>, rows> evaluations;
        std::array<std::vector<double>, rows> vertices;
        std::array<std::vector<double>, rows> primarySamples;
        std::array<std::vector<double>, rows> productiveSamples;
        std::array<std::vector<double>, rows> verticesPerSample;
        std::array<std::vector<double>, rows> pathLength;
        std::array<std::vector<double>, rows> radPerCall;
        std::array<std::vector<double>, rows> coarse;
        std::array<std::vector<double>, rows> hopFloored, hopAtRegion, hopEdgeLimited;
        /// Per-problem QP program size and solver-hit rate, pushed only by rows that
        /// build a QP, so a row without one reports an empty distribution rather than a
        /// zero that reads as "assembled nothing" next to rows that assembled plenty.
        std::array<std::vector<double>, rows> qpRowsPerCall, qpSolveShare;
        std::array<std::size_t, rows> qpCalls{};
        std::array<std::size_t, rows> unsafe{};
        std::array<std::size_t, rows> audited{};
        std::array<std::size_t, rows> misses{};
        std::array<std::size_t, rows> selfColliding{};
        std::array<double, rows> worstClearance;
        std::array<double, rows> worstSelf;

        Tally()
        {
            worstClearance.fill(std::numeric_limits<double>::infinity());
            worstSelf.fill(std::numeric_limits<double>::infinity());
        }

        void add(int row, const Result &result)
        {
            solved[row] += result.solved ? 1 : 0;
            seconds[row].push_back(result.seconds);
            evaluations[row].push_back(static_cast<double>(result.evaluations));
            vertices[row].push_back(static_cast<double>(result.vertices));
            primarySamples[row].push_back(static_cast<double>(result.primarySamples));
            productiveSamples[row].push_back(static_cast<double>(result.productiveSamples));
            verticesPerSample[row].push_back(
                result.primarySamples > 0
                    ? static_cast<double>(result.vertices) / result.primarySamples
                    : 0.0);
            if (result.solved)
                pathLength[row].push_back(result.pathLength);
            radPerCall[row].push_back(result.radPerCall);
            coarse[row].push_back(result.coarse);
            hopFloored[row].push_back(result.hopFloored);
            hopAtRegion[row].push_back(result.hopAtRegion);
            hopEdgeLimited[row].push_back(result.hopEdgeLimited);
            qpCalls[row] += result.qpCalls;
            if (result.qpCalls > 0)
            {
                const double calls = static_cast<double>(result.qpCalls);
                qpRowsPerCall[row].push_back(static_cast<double>(result.qpRows) / calls);
                qpSolveShare[row].push_back(static_cast<double>(result.qpSolves) / calls);
            }
            unsafe[row] += result.unsafeStates;
            audited[row] += result.auditedStates;
            misses[row] += result.misses;
            selfColliding[row] += result.selfColliding;
            if (result.solved)
            {
                worstClearance[row] = std::min(worstClearance[row], result.minClearance);
                worstSelf[row] = std::min(worstSelf[row], result.minSelfOverlap);
            }
        }
    };

    double median(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    /// min/median/max in one sort, for reporting a timing *distribution* rather
    /// than a single point estimate -- issue 9 asks for the spread, not just the
    /// middle.
    struct Distribution
    {
        double min{0.0};
        double median{0.0};
        double max{0.0};
    };

    Distribution distribution(std::vector<double> values)
    {
        if (values.empty())
            return {};
        std::sort(values.begin(), values.end());
        return {values.front(), values[values.size() / 2], values.back()};
    }

    /// Append one problem's audited motion, preceded by the marker the Python auditor
    /// splits on. The marker matters: the file holds many unrelated motions, and
    /// interpolating across the seam between two of them invents states that belong to no
    /// trajectory the planner produced -- which on a self-collision count reads as the
    /// planner having folded the arm onto itself.
    void writeMotion(std::ofstream &out, const Problem &problem,
                     const std::vector<UR5::Configuration> &states)
    {
        if (!out.is_open() || states.empty())
            return;
        out << "# motion " << problem.scene << " " << problem.index << "\n";
        for (const UR5::Configuration &q : states)
        {
            for (int j = 0; j < dimension; ++j)
                out << (j ? " " : "") << q[j];
            out << "\n";
        }
    }

    /// \p name is the `OMPL_MBM_ROWS` name of the row, so a deselected row prints
    /// nothing rather than a line of zeros indistinguishable from total failure.
    void reportRow(const char *label, const char *name, const Tally &tally, int row)
    {
        if (!wantRow(name))
            return;
        const int scored = static_cast<int>(tally.seconds[row].size());
        const Distribution ms = distribution(tally.seconds[row]);
        std::printf("  %-11s %3d/%-4d %7.2f/%.2f/%-7.2f %10.0f %8.0f %8.2f", label, tally.solved[row],
                    scored, 1e3 * ms.min, 1e3 * ms.median, 1e3 * ms.max,
                    median(tally.evaluations[row]), median(tally.vertices[row]),
                    median(tally.pathLength[row]));
        if (median(tally.radPerCall[row]) > 0.0)
            std::printf(" %8.4f %5.0f%%", median(tally.radPerCall[row]),
                        1e2 * median(tally.coarse[row]));
        else
            std::printf(" %8s %6s", "-", "-");
        if (tally.solved[row] > 0)
            std::printf(" %10.4f %6zu/%-7zu %6zu %10.4f %7zu\n", tally.worstClearance[row],
                        tally.unsafe[row], tally.audited[row], tally.misses[row],
                        tally.worstSelf[row], tally.selfColliding[row]);
        else
            std::printf(" %10s %14s %6s %10s %7s\n", "-", "-", "-", "-", "-");
        const double primary = median(tally.primarySamples[row]);
        const double productive = median(tally.productiveSamples[row]);
        std::printf("      primary samples %.0f, productive %.0f (%.1f%%), vertices/sample %.3f\n",
                    primary, productive, primary > 0.0 ? 1e2 * productive / primary : 0.0,
                    median(tally.verticesPerSample[row]));
        // Medians of per-problem ratios, not ratios of totals: a single hard problem
        // makes far more filter calls than an easy one, and a pooled ratio would report
        // that problem's QP rather than the row's.
        if (tally.qpCalls[row] > 0)
            std::printf("      qp %.1f rows/call, %.0f%% of calls reached the solver,"
                        " %zu calls\n",
                        median(tally.qpRowsPerCall[row]),
                        1e2 * median(tally.qpSolveShare[row]), tally.qpCalls[row]);
    }

    void writeCsvRow(std::ofstream &out, unsigned long seed, const Problem &problem,
                     const char *method, bool eligible, const Result &result)
    {
        if (!out.is_open() || !wantRow(method))
            return;
        out << seed << ',' << problem.scene << ',' << problem.index << ',' << method << ','
            << (eligible ? 1 : 0) << ',' << (result.solved ? 1 : 0) << ',' << result.seconds << ','
            << result.evaluations << ',' << result.vertices << ',' << result.pathLength << ','
            << result.waypoints << ',' << result.auditedStates << ',' << result.unsafeStates << ','
            << result.minClearance << ',' << result.minSelfOverlap << ',' << result.selfColliding << ','
            << result.misses << ',' << result.radPerCall << ',' << result.coarse << ','
            << result.primarySamples << ',' << result.productiveSamples << ','
            << (result.primarySamples > 0
                    ? static_cast<double>(result.vertices) / result.primarySamples
                    : 0.0)
            << ',' << result.picardAttempts << ',' << result.picardAccepted << ','
            << result.picardFallbacks << ',' << result.hopFloored << ','
            << result.hopAtRegion << ',' << result.hopEdgeLimited << ',' << result.qpCalls
            << ',' << result.qpRows << ',' << result.qpSolves << '\n';
    }
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s scenes.txt [perScene] [seconds] [voxel] [stepSize] [range]\n"
                    "       [margin] [buffer] [segmentFraction] [kappa] [maxStepScale]\n"
                    "       [selfMargin] [pathPrefix] [shortcutDelta] [seed] [csvPath] [safeHops]\n"
                    "       [picardIterations] [picardWindow] [picardWorkers]"
                    " [trajectoryPrefixes] [rolloutCallBudget]\n\n"
                    "Generate scenes.txt with scripts/mbm_to_scenes.py.\n",
                    argv[0]);
        return 1;
    }

    // The L1 baseline is the *arm-length* table, not the swept-enclosure-tightened
    // lever-arm one. Both bound `|dp_i/dq_k|`; arm length bounds distance-to-axis by
    // full distance-to-origin, discarding the axial component, and is what the
    // published weighted-L1 certificate uses. The tightening is this repository's own
    // improvement, so folding it into the baseline would compare the envelope against
    // a stronger L1 than anything in the literature and understate the gap.
    //
    // Set before any table is touched, and only if the caller has not chosen: pass
    // OMPL_UR5_LEVER_BOUNDS=lever_arm to measure against the tightened bound instead.
    // `qpEnvelope` is unaffected either way -- it reads the same table for its own L1
    // term and scores identically on both -- so this moves `qpAdaptive` alone.
    ::setenv("OMPL_UR5_LEVER_BOUNDS", "arm_length", 0);
    ::setenv("OMPL_ROBOT_LEVER_BOUNDS", "arm_length", 0);

    const std::string path = argv[1];
    const int perScene = argc > 2 ? std::atoi(argv[2]) : 10;
    const double timeLimit = argc > 3 ? std::atof(argv[3]) : 5.0;
    const double voxel = argc > 4 ? std::atof(argv[4]) : 0.03;
    const double stepSize = argc > 5 ? std::atof(argv[5]) : 0.01;
    const double range = argc > 6 ? std::atof(argv[6]) : 2.0;
    // The audited margin, and the extra the filter guards on top of it. MotionBenchMaker
    // endpoints are grasp poses sitting ~8 mm off the shelf, so the defaults
    // (0.06 + one voxel) rule out most of the set -- these exist to find out what does
    // fit, with the unsafe column as the check on having shrunk them too far.
    const double margin = argc > 7 ? std::atof(argv[7]) : Barrier::defaultMargin;
    const double buffer = argc > 8 ? std::atof(argv[8]) : -1.0;
    // Resolution the baseline checks its straight-line edges at, as a fraction of the
    // state space's maximum extent. OMPL defaults to 0.01, which on this space is 0.31
    // rad between samples -- far coarser than the audit, so the baseline is scored unsafe
    // at the default. Tightening this is what makes the comparison like for like: both
    // rows then have to be audit-clean, and the cost of being so is the thing to compare.
    const double segmentFractionArg = argc > 9 ? std::atof(argv[9]) : -1.0;
    // The CBF decay rate in 1/s: `dh/dt >= -kappa h`, so it sets how fast clearance may
    // be spent and scales the certified step directly. Large values approach a plain
    // collision condition, with the certificate as long as the clearance allows. The
    // default is what the old per-step `gamma 0.4` amounted to at a 0.05 s step.
    const double kappa = argc > 10 ? std::atof(argv[10]) : 8.0;
    // Cap on the certified step, as a multiple of stepSize. 1.0 is the fixed-step A/B.
    const double maxStepScale = argc > 11 ? std::atof(argv[11]) : -1.0;
    // The self-collision margin. A large negative value is the A/B for the rows
    // themselves: every pair clearance becomes enormous, so none is ever screened in and
    // none can cap the certificate, which is what the barrier did before it modelled the
    // arm against itself. The independent audit column is unaffected either way, so it
    // says what turning them off costs.
    const double selfMargin = argc > 12 ? std::atof(argv[12]) : Barrier::defaultSelfMargin;
    // Where to dump the audited motions, one file per row (`<prefix>.cbf`, `<prefix>.rrtc`).
    // The `collide` column is a statement about the 40 spheres and nothing else; these
    // files are what lets ur5_experiments/scripts/audit_self_collision.py repeat the count
    // against the real meshes, which is the only thing that can say the sphere model is
    // too coarse to stand in for them.
    const std::string pathPrefix = argc > 13 ? argv[13] : std::string();
    // Rope anchor spacing, in radians, applied to all rows at the same value after they
    // solve and before they are audited. Non-positive (the default) leaves both
    // unsimplified -- shortcutting one row and not the other measures rope's
    // aggressiveness, not the filter.
    const double shortcutDelta = argc > 14 ? std::atof(argv[14]) : -1.0;
    const unsigned long seed = argc > 15 ? std::strtoul(argv[15], nullptr, 10) : 1UL;
    const std::string csvPath = argc > 16 ? argv[16] : std::string();
    // Whether a rollout hop may spend the safety certificate rather than the no-op one --
    // `ClearanceBarrier::safeDuration()` against `certifiedDuration()`. Longer hops and
    // fewer filter calls, at the cost of the edge no longer being the filtered edge.
    // On by default, matching `FilteredStateSpace`; zero recovers the behaviour every
    // run before the certified region had.
    const bool safeHops = argc > 17 ? std::atoi(argv[17]) != 0 : true;
    (void)safeHops;  // retained for command-line compatibility; the A/B runs both modes
    const unsigned int picardIterations =
        argc > 18 ? static_cast<unsigned int>(std::max(0, std::atoi(argv[18]))) : 0u;
    const unsigned int picardWindow =
        argc > 19 ? static_cast<unsigned int>(std::max(1, std::atoi(argv[19]))) : 4u;
    const unsigned int picardWorkers =
        argc > 20 ? static_cast<unsigned int>(std::max(0, std::atoi(argv[20]))) : 0u;
    const unsigned int trajectoryPrefixes =
        argc > 21 ? static_cast<unsigned int>(std::max(0, std::atoi(argv[21]))) : 0u;
    const unsigned int rolloutCallBudget =
        argc > 22 ? static_cast<unsigned int>(std::max(0, std::atoi(argv[22]))) : 40u;

    // Tie the baseline's edge-checking spacing to the rollout's step unless told otherwise,
    // so neither row is scored at a resolution the other never saw. The rollout advances
    // `maxSpeed * stepSize` radians per filter call; OMPL states the baseline's resolution
    // as a fraction of the state space's maximum extent, so convert. A RealVectorStateSpace
    // over `dimension` joints spanning [lo, hi] has extent |hi - lo| * sqrt(dimension).
    const double rolloutStep = effectiveMaxSpeed().maxCoeff() * stepSize;
    const double extent =
        (UR5::upperBounds() - UR5::lowerBounds()).norm();
    const double segmentFraction =
        segmentFractionArg > 0.0 ? segmentFractionArg : rolloutStep / extent;

    // Split the problem set over independent processes: `OMPL_MBM_SHARD=i/n` runs the
    // problems whose ordinal is `i` mod `n`. Each problem keeps the seed it would have
    // had in a single-process run, so concatenating the shards' CSVs reproduces the
    // unsharded CSV exactly -- only the per-shard summary tables are partial, since each
    // process can only tally what it ran. Sharding buys parallel *planning*, which
    // `bakeConcurrent` does not: at a fine voxel one field is ~3.7 GB, so the practical
    // ceiling is memory, roughly a dozen shards in 62 GB.
    unsigned shardIndex = 0u;
    unsigned shardCount = 1u;
    if (const char *v = std::getenv("OMPL_MBM_SHARD"))
    {
        if (std::sscanf(v, "%u/%u", &shardIndex, &shardCount) != 2 || shardCount == 0u ||
            shardIndex >= shardCount)
        {
            std::fprintf(stderr, "OMPL_MBM_SHARD must read i/n with 0 <= i < n, got '%s'\n", v);
            return 1;
        }
    }
    // Threads for the field bake. Defaults to the hardware count, which is right for one
    // process and oversubscribes when several shards run at once -- set it to
    // cores/shards in that case.
    unsigned bakeThreads = 0u;
    if (const char *v = std::getenv("OMPL_SDF_BAKE_THREADS"))
        bakeThreads = static_cast<unsigned>(std::max(0, std::atoi(v)));

    ompl::RNG::setSeed(seed);
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);

    const std::vector<Problem> problems = readProblems(path);
    const UR5 robot;

    Filter::Parameters parameters;
    parameters.kappa = kappa;
    parameters.maxSpeed = effectiveMaxSpeed();
    parameters.respectJointLimits = true;
    // The geometric state space already bounds sampled configurations. This switch
    // isolates the effect of additionally constraining one Euler step inside the QP.
    if (const char *v = std::getenv("OMPL_CBF_JOINT_LIMITS"))
        parameters.respectJointLimits = std::atoi(v) != 0;
    // On for the QP rows -- `qpFixed`, `qpAdaptive` and `qpEnvelope`. `qpPlain` keeps it
    // off along with the rest of the QP-side work, so it stays the unassisted reference.
    //
    // Unlike screening this is *not* solution-preserving: it caps the certificate at
    // `pairRelevance * max(dt, 1/kappa)`, so it moves the certificate rows' results and
    // not merely their speed. Against the arm-length L1 table the cap barely bites
    // (`qpAdaptive` 1,523 to 1,525); against the tightened lever-arm table it does
    // (1,482 to 1,495), so a sweep using it has to say which L1 it ran. It is sound with
    // the envelope: a dropped pair is proven clear for `relevance * scale` while
    // `EnvelopeHoldFilter` clips at `1 * scale`, which is conservative.
    //
    // `qpFreeGate` is unaffected either way -- it never touches the QP solver, so it
    // reads none of this.
    parameters.activePairs = true;
    // Process-isolated ablation switch for the active-set pair traversal, in the style
    // of OMPL_UR5_LEVER_BOUNDS. See ClearanceBarrier::ActiveSet.
    if (const char *v = std::getenv("OMPL_CBF_ACTIVE_PAIRS"))
        parameters.activePairs = std::atoi(v) != 0;
    if (const char *v = std::getenv("OMPL_CBF_PAIR_RELEVANCE"))
        parameters.pairRelevance = std::atof(v);
    // Widens the QP's row screen, and with it the horizon any certificate built from
    // the screened set may claim. Costs QP rows; see Parameters::screenHorizonScale.
    if (const char *v = std::getenv("OMPL_CBF_SCREEN_SCALE"))
        parameters.screenHorizonScale = std::atof(v);
    // Off by default *here*, unlike the library. Screening is sound and a pure speed
    // win -- it keeps the QP's answer identical -- but it makes a filter call about
    // half as expensive, and a hold certificate's cost is fixed per call while its
    // benefit is proportional to what a call costs. Benchmarking the certificates
    // against a screened filter therefore measures them in their least favourable
    // regime: the same rows read -2.5% against `qpFixed` screened and -16.3%
    // unscreened. Unscreened is the honest denominator for a certificate study, and
    // it also removes the clip the screened path imposes on the certificate. The
    // library default is untouched; set OMPL_CBF_SCREENING=1 to restore it.
    parameters.screening = false;
    if (const char *v = std::getenv("OMPL_CBF_SCREENING"))
        parameters.screening = std::atoi(v) != 0;
    // Solve the QP with no shortcuts -- no feasibility bypass, no closed-form one-row
    // projection, no pre-inverted Cholesky factor. See Parameters::plainSolve.
    if (const char *v = std::getenv("OMPL_CBF_PLAIN_QP"))
        parameters.plainSolve = std::atoi(v) != 0;
    // Isolate the one-row projection from the other QP shortcuts.
    if (const char *v = std::getenv("OMPL_CBF_CLOSED_FORM_PROJECTION"))
        parameters.closedFormProjection = std::atoi(v) != 0;

    Filter::Parameters fixedParameters = parameters;
    fixedParameters.certificates = false;

    // The unassisted reference: `qpFixed`'s controller -- fixed step, no hold certificate
    // -- solved by qpmad with none of this repository's QP-side work. `plainSolve` drops
    // the feasibility bypass, the closed-form one-row projection and the pre-inverted
    // Cholesky factor; `screening` drops the row screen, so every sphere and pair
    // contributes a constraint whether or not it could bind; `activePairs` drops the
    // active-set pair traversal.
    //
    // This is what a straightforward CBF-QP filter looks like, and it is the bottom rung
    // of the ladder the other rows climb. It prices all three together on purpose -- read
    // `qpPlain -> qpFixed` as "our QP-side work", and use OMPL_CBF_PLAIN_QP,
    // OMPL_CBF_SCREENING and OMPL_CBF_ACTIVE_PAIRS to separate them.
    Filter::Parameters plainParameters = fixedParameters;
    plainParameters.plainSolve = true;
    plainParameters.screening = false;
    plainParameters.activePairs = false;

    std::printf("\nMotionBenchMaker UR5, %d problems loaded, up to %d per scene\n",
                static_cast<int>(problems.size()), perScene);
    std::printf("voxel %.3f m, margin %.4f m + %.4f m filter buffer, stepSize %.3f s, "
                "range %.2f rad, %.1f s limit\n\n",
                voxel, margin,
                buffer < 0.0 ? Barrier::interpolationBuffer(
                                   sdf::GridSDF::bakeConcurrent(problems.front().field(),
                                                                UR5::reachableBounds(), voxel))
                             : buffer,
                stepSize, range, timeLimit);
    std::printf("hop certificates: old L1 Lipschitz region and new hold-time bound\n");
    if (picardIterations > 0)
        std::printf("Picard: %u maps, %u-step windows, %u workers; ", picardIterations,
                    picardWindow, picardWorkers);
    else
        std::printf("Picard: off; ");
    std::printf("trajectory prefixes: %u; sequential rollout budget: %u calls\n",
                trajectoryPrefixes, rolloutCallBudget);
    std::printf("kappa %.2f /s, certified step %s, self-collision margin %.4f m over %d pairs%s\n",
                kappa, maxStepScale > 0.0 ? "capped" : "uncapped", selfMargin,
                static_cast<int>(UR5::nSelfPairs), selfMargin < 0.0 ? " (rows disabled)" : "");
    std::printf("baseline segment: %.6f of extent = %.4f rad, rollout step %.4f rad (%s)\n",
                segmentFraction, segmentFraction * extent, rolloutStep,
                segmentFractionArg > 0.0 ? "overridden" : "matched to the rollout");
    if (shortcutDelta > 0.0)
        std::printf("rope shortcut on, delta %.4f rad, all rows\n\n", shortcutDelta);
    else
        std::printf("rope shortcut off, all rows unsimplified\n\n");
    std::printf("seed %lu\n", seed);
    std::printf("  %-11s %8s %-19s %10s %8s %8s %8s %6s %10s %14s %6s %10s %7s\n", "planner", "solved",
                "ms(min/med/max)", "samples", "vertices", "path", "rad/call", "coarse", "worst clr",
                "unsafe/audited", "missed", "worst self", "collide");

    std::map<std::string, Tally> tallies;
    std::map<std::string, int> seen;
    Tally overall;

    std::ofstream csvOut;
    if (!csvPath.empty())
    {
        csvOut.open(csvPath);
        if (!csvOut.is_open())
            throw ompl::Exception("cannot write " + csvPath);
        csvOut << "seed,scene,problem,method,eligible,solved,seconds,samples,vertices,path_length,"
                  "waypoints,audited_states,unsafe_states,min_clearance,min_self_overlap,"
                  "self_colliding,misses,rad_per_call,coarse_fraction,primary_samples,"
                  "productive_samples,vertices_per_sample,picard_attempts,picard_accepted,"
                  "picard_fallbacks,hop_floored,hop_at_region,hop_edge_limited,"
                  "qp_calls,qp_rows,qp_solves\n";
    }

    std::ofstream baselineOut, fixedOut, filteredOut, gateOut, vampOut;
    if (!pathPrefix.empty())
    {
        baselineOut.open(pathPrefix + ".rrtc");
        fixedOut.open(pathPrefix + ".qp-fixed");
        filteredOut.open(pathPrefix + ".cbf");
        gateOut.open(pathPrefix + ".qp-free");
#ifdef OMPL_MBM_HAVE_VAMP
        vampOut.open(pathPrefix + ".vamp");
#endif
        if (!baselineOut.is_open() || !fixedOut.is_open() || !filteredOut.is_open() ||
            !gateOut.is_open()
#ifdef OMPL_MBM_HAVE_VAMP
            || !vampOut.is_open()
#endif
        )
        {
            std::printf("cannot write %s.{rrtc,qp-fixed,cbf,qp-free}\n", pathPrefix.c_str());
            return 1;
        }
        for (std::ofstream *out : {&baselineOut, &fixedOut, &filteredOut, &gateOut
#ifdef OMPL_MBM_HAVE_VAMP
                                   , &vampOut
#endif
             })
            *out << "# audited joint-space motions, one configuration per line, "
                 << auditResolutionValue() << " rad spacing\n"
                 << "# each motion begins with a '# motion <scene> <index>' marker\n";
    }

    // Position in the filtered problem sequence, 1-based -- what `overall.attempted`
    // counts in an unsharded run. Kept separately so it stays the problem's *absolute*
    // ordinal when only a shard of the set runs in this process, which is what makes
    // `sampleSeed` below independent of the sharding.
    unsigned long ordinal = 0;

    for (const Problem &problem : problems)
    {
        if (seen[problem.scene]++ >= perScene)
            continue;

        // Every problem that clears the per-scene filter takes the next ordinal, whether
        // or not this process is the one that runs it.
        ++ordinal;
        if (shardCount > 1u && (ordinal - 1u) % shardCount != shardIndex)
            continue;

        // Threaded bake: bit-identical to the serial one, but a 3 mm grid is 463 M
        // nodes and 11.2 s single-threaded, paid once per problem. See bakeConcurrent.
        const sdf::GridSDF field =
            sdf::GridSDF::bakeConcurrent(problem.field(), UR5::reachableBounds(), voxel,
                                         bakeThreads);
        // The barrier the assertions use, and the thicker one the filter guards so that
        // auditing against the first one passes. See ClearanceBarrier::guarding().
        const Barrier audited(robot, field, margin, selfMargin);
        const Barrier guard =
            Barrier::guarding(robot, field, margin,
                              buffer < 0.0 ? Barrier::interpolationBuffer(field) : buffer,
                              selfMargin);
        const Filter fixedFilter(guard, fixedParameters);
        const Filter plainFilter(guard, plainParameters);
        const Filter lipschitzFilter(guard, parameters);
        const ompl::demo::UR5QPFreeGate qpFreeGate(guard, parameters);
        // Row 5 of the planner comparison: the same QP as `qpAdaptive`, hopping on the
        // motion-envelope certificate instead of the L1 lever-arm one. One object apart.
        const ompl::demo::EnvelopeHoldFilter envelopeFilter(guard, parameters);
        const ompl::cbf::CertifiedRegionRollout oldCertificate(guard);
        const HoldTimeRegionRollout newCertificate(guard, robot, true);
        const Space::RolloutPlanner oldCertificatePlanner = oldCertificate.planner();
        const Space::RolloutPlanner newCertificatePlanner = newCertificate.planner();

        Tally &tally = tallies[problem.scene];
        ++tally.attempted;
        ++overall.attempted;
        // The problem's absolute ordinal, not this process's running count, so a sharded
        // run reproduces an unsharded one problem for problem.
        const std::uint_fast32_t sampleSeed = static_cast<std::uint_fast32_t>(seed + ordinal);

        const Barrier bare(robot, field, 0.0, selfMargin);
        const double endpoints =
            std::min(bare.worstValue(problem.start), bare.worstValue(problem.goal));
        tally.endpointClearance.push_back(endpoints);
        overall.endpointClearance.push_back(endpoints);

        // Valid upstream does not mean feasible here: our spheres do not enclose the
        // meshes MotionBenchMaker checked, and the margin sits on top of that. Split by
        // which barrier rejected the endpoint -- world clearance and self-collision are
        // different failure modes with different fixes, see Tally::skippedClearance.
        //
        // Tested against the barrier the *filter* enforces, not the one the audit uses.
        // Those differ by the interpolation buffer, and the difference is not academic:
        // a MotionBenchMaker goal is a grasp pose about 8 mm off the shelf, the buffer at
        // a 30 mm voxel is 30 mm, and `RRTConnect` roots its second tree at the goal. A
        // CBF gives forward invariance of {h >= 0} starting *from* {h >= 0}; rooted at
        // h < 0 it guarantees nothing, and every rollout grown from that tree inherits
        // the violation. Screening endpoints at the audited margin instead let those
        // problems through and is what put real penetration in the CBF rows while the
        // skip count read zero.
        const double guardBuffer = buffer < 0.0 ? Barrier::interpolationBuffer(field) : buffer;
        const Barrier worldOnly(robot, field, margin + guardBuffer, -1e6);
        const Barrier selfOnly(robot, field, -1e6, selfMargin + Barrier::defaultSelfBuffer);
        const bool worldUnsafe = !worldOnly.isSafe(problem.start) || !worldOnly.isSafe(problem.goal);
        const bool selfUnsafe = !selfOnly.isSafe(problem.start) || !selfOnly.isSafe(problem.goal);
        if (worldUnsafe || selfUnsafe)
        {
            ++tally.skipped;
            ++overall.skipped;
            if (worldUnsafe)
            {
                ++tally.skippedClearance;
                ++overall.skippedClearance;
            }
            if (selfUnsafe)
            {
                ++tally.skippedSelfCollision;
                ++overall.skippedSelfCollision;
            }
            const Result skipped;
            writeCsvRow(csvOut, seed, problem, "isSafe", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpFixed", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpPlain", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpAdaptive", false, skipped);
            writeCsvRow(csvOut, seed, problem, "l1Old", false, skipped);
            writeCsvRow(csvOut, seed, problem, "holdNew", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpFreeGate", false, skipped);
            writeCsvRow(csvOut, seed, problem, "qpEnvelope", false, skipped);
#ifdef OMPL_MBM_HAVE_VAMP
            writeCsvRow(csvOut, seed, problem, "VAMP", false, skipped);
#endif
            continue;
        }

        std::vector<UR5::Configuration> checkedPath, fixedPath, lipschitzPath, rolledPath, gatePath;
        const Result checked =
            wantRow("isSafe") ? runCollisionChecked(problem, audited, range, timeLimit,
                                                    segmentFraction, shortcutDelta, sampleSeed,
                                                    pathPrefix.empty() ? nullptr : &checkedPath)
                              : Result();
        const Result fixed =
            wantRow("qpFixed")
                ? runFiltered(problem, audited, fixedFilter, stepSize, range, timeLimit, 1.0,
                              shortcutDelta, false, picardIterations, picardWindow, picardWorkers,
                              trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              pathPrefix.empty() ? nullptr : &fixedPath)
                : Result();
        // Identical to qpFixed in every argument; only the filter's parameters differ,
        // so the pair isolates this repository's QP-side work.
        const Result plain =
            wantRow("qpPlain")
                ? runFiltered(problem, audited, plainFilter, stepSize, range, timeLimit, 1.0,
                              shortcutDelta, false, picardIterations, picardWindow, picardWorkers,
                              trajectoryPrefixes, rolloutCallBudget, sampleSeed, nullptr)
                : Result();
        // Same QP controller as qpFixed, but allow the filter's certified duration
        // to hold the solved control beyond one integration step.
        const Result qpAdaptive =
            wantRow("qpAdaptive")
                ? runFiltered(problem, audited, lipschitzFilter, stepSize, range, timeLimit,
                              maxStepScale, shortcutDelta, true, picardIterations, picardWindow,
                              picardWorkers, trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              nullptr)
                : Result();
        const Result oldLipschitz =
            wantRow("l1Old")
                ? runFiltered(problem, audited, lipschitzFilter, stepSize, range, timeLimit,
                              maxStepScale, shortcutDelta, false, picardIterations, picardWindow,
                              picardWorkers, trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              pathPrefix.empty() ? nullptr : &lipschitzPath, &oldCertificatePlanner)
                : Result();
        const Result rolled =
            wantRow("holdNew")
                ? runFiltered(problem, audited, lipschitzFilter, stepSize, range, timeLimit,
                              maxStepScale, shortcutDelta, true, picardIterations, picardWindow,
                              picardWorkers, trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              pathPrefix.empty() ? nullptr : &rolledPath, &newCertificatePlanner)
                : Result();
        const Result envelope =
            wantRow("qpEnvelope")
                ? runFiltered(problem, audited, envelopeFilter, stepSize, range, timeLimit,
                              maxStepScale, shortcutDelta, true, picardIterations, picardWindow,
                              picardWorkers, trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              nullptr)
                : Result();
        const Result gated =
            wantRow("qpFreeGate")
                ? runFiltered(problem, audited, qpFreeGate, stepSize, range, timeLimit, 1.0,
                              shortcutDelta, false, 0, picardWindow, picardWorkers,
                              trajectoryPrefixes, rolloutCallBudget, sampleSeed,
                              pathPrefix.empty() ? nullptr : &gatePath)
                : Result();
#ifdef OMPL_MBM_HAVE_VAMP
        std::vector<UR5::Configuration> vampPath;
        const Result vamp =
            wantRow("VAMP") ? runVamp(problem, audited, range, timeLimit, shortcutDelta, sampleSeed,
                                      pathPrefix.empty() ? nullptr : &vampPath)
                            : Result();
#endif
        writeMotion(baselineOut, problem, checkedPath);
        writeMotion(fixedOut, problem, fixedPath);
        writeMotion(filteredOut, problem, rolledPath);
        writeMotion(gateOut, problem, gatePath);
        tally.add(checkedRow, checked);
        tally.add(qpFixedRow, fixed);
        tally.add(qpPlainRow, plain);
        tally.add(qpAdaptiveRow, qpAdaptive);
        tally.add(qpLipschitzRow, oldLipschitz);
        tally.add(qpSafeRow, rolled);
        tally.add(qpFreeRow, gated);
        tally.add(qpEnvelopeRow, envelope);
        overall.add(checkedRow, checked);
        overall.add(qpFixedRow, fixed);
        overall.add(qpPlainRow, plain);
        overall.add(qpAdaptiveRow, qpAdaptive);
        overall.add(qpLipschitzRow, oldLipschitz);
        overall.add(qpSafeRow, rolled);
        overall.add(qpFreeRow, gated);
        overall.add(qpEnvelopeRow, envelope);
        writeCsvRow(csvOut, seed, problem, "isSafe", true, checked);
        writeCsvRow(csvOut, seed, problem, "qpFixed", true, fixed);
        writeCsvRow(csvOut, seed, problem, "qpPlain", true, plain);
        writeCsvRow(csvOut, seed, problem, "qpAdaptive", true, qpAdaptive);
        writeCsvRow(csvOut, seed, problem, "l1Old", true, oldLipschitz);
        writeCsvRow(csvOut, seed, problem, "holdNew", true, rolled);
        writeCsvRow(csvOut, seed, problem, "qpFreeGate", true, gated);
        writeCsvRow(csvOut, seed, problem, "qpEnvelope", true, envelope);
#ifdef OMPL_MBM_HAVE_VAMP
        tally.add(vampRow, vamp);
        overall.add(vampRow, vamp);
        writeMotion(vampOut, problem, vampPath);
        writeCsvRow(csvOut, seed, problem, "VAMP", true, vamp);
#endif
    }

    if (!pathPrefix.empty())
    {
        baselineOut.close();
        fixedOut.close();
        filteredOut.close();
        gateOut.close();
        std::printf("wrote %s.{rrtc,qp-fixed,cbf,qp-free,vamp} -- audit them against the meshes with\n"
                    "  ur5_experiments/scripts/audit_self_collision.py --path %s.cbf\n",
                    pathPrefix.c_str(), pathPrefix.c_str());
    }

    for (const auto &entry : tallies)
    {
        const Tally &tally = entry.second;
        std::printf("\n%s  (%d problems, %d skipped: %d clearance, %d self-collision)\n",
                    entry.first.c_str(), tally.attempted, tally.skipped, tally.skippedClearance,
                    tally.skippedSelfCollision);
        reportRow("rrtconnect", "isSafe", tally, checkedRow);
        reportRow("qp-fixed", "qpFixed", tally, qpFixedRow);
        reportRow("qp-plain", "qpPlain", tally, qpPlainRow);
        reportRow("qp-adapt", "qpAdaptive", tally, qpAdaptiveRow);
        reportRow("l1-old", "l1Old", tally, qpLipschitzRow);
        reportRow("hold-new", "holdNew", tally, qpSafeRow);
        reportRow("qp-free", "qpFreeGate", tally, qpFreeRow);
        reportRow("qp-env", "qpEnvelope", tally, qpEnvelopeRow);
#ifdef OMPL_MBM_HAVE_VAMP
        reportRow("vamp-rrtc", "VAMP", tally, vampRow);
#endif
    }

    // The feasibility question, since it decides how much of the benchmark is usable.
    // The question a longer certificate has to answer: is the hop ending at the edge of
    // the certified region, or is something else stopping it first? Only hops in the
    // first column can be converted by a better certificate -- a hop that already runs
    // to the region's edge, or to the end of its extension, is not certificate-limited.
    {
        bool any = false;
        for (const auto &entry : tallies)
            for (int row = 0; row < comparisonRows; ++row)
                any = any || !entry.second.hopAtRegion[row].empty();
        if (any)
        {
            std::printf("\nHop disposition -- what set the hop's length, median over problems.\n");
            std::printf("  %-11s %14s %12s %14s\n", "row", "floored(<1 step)", "at region",
                        "edge-limited");
            const auto line = [&](const char *label, const char *name, int row)
            {
                if (!wantRow(name) || overall.hopAtRegion[row].empty())
                    return;
                std::printf("  %-11s %13.1f%% %11.1f%% %13.1f%%\n", label,
                            1e2 * median(overall.hopFloored[row]),
                            1e2 * median(overall.hopAtRegion[row]),
                            1e2 * median(overall.hopEdgeLimited[row]));
            };
            line("qp-fixed", "qpFixed", qpFixedRow);
            line("qp-plain", "qpPlain", qpPlainRow);
            line("qp-adapt", "qpAdaptive", qpAdaptiveRow);
            line("l1-old", "l1Old", qpLipschitzRow);
            line("hold-new", "holdNew", qpSafeRow);
            line("qp-free", "qpFreeGate", qpFreeRow);
            line("qp-env", "qpEnvelope", qpEnvelopeRow);
        }
    }

    std::printf("\nEndpoint clearance at zero margin -- min(start, goal) over all spheres.\n");
    std::printf("  %-20s %8s %9s %9s %9s   %s\n", "scene", "min", "median", "max",
                "affordable", "problems by margin");
    for (const auto &entry : tallies)
    {
        std::vector<double> clearances = entry.second.endpointClearance;
        if (clearances.empty())
            continue;
        std::sort(clearances.begin(), clearances.end());
        const auto share = [&clearances](double margin)
        {
            std::size_t n = 0;
            for (const double c : clearances)
                n += (c >= margin) ? 1 : 0;
            return 100.0 * static_cast<double>(n) / static_cast<double>(clearances.size());
        };
        std::printf("  %-20s %8.4f %9.4f %9.4f %9.4f   0:%3.0f%% 0.02:%3.0f%% 0.04:%3.0f%% "
                    "0.06:%3.0f%% 0.09:%3.0f%%\n",
                    entry.first.c_str(), clearances.front(), median(clearances), clearances.back(),
                    clearances.front(), share(0.0), share(0.02), share(0.04), share(0.06),
                    share(0.09));
    }

    std::printf("\nall scenes  (%d problems, %d skipped: %d clearance, %d self-collision)\n",
                overall.attempted, overall.skipped, overall.skippedClearance,
                overall.skippedSelfCollision);
    reportRow("rrtconnect", "isSafe", overall, checkedRow);
    reportRow("qp-fixed", "qpFixed", overall, qpFixedRow);
    reportRow("qp-plain", "qpPlain", overall, qpPlainRow);
    reportRow("qp-adapt", "qpAdaptive", overall, qpAdaptiveRow);
    reportRow("l1-old", "l1Old", overall, qpLipschitzRow);
    reportRow("hold-new", "holdNew", overall, qpSafeRow);
    reportRow("qp-free", "qpFreeGate", overall, qpFreeRow);
    reportRow("qp-env", "qpEnvelope", overall, qpEnvelopeRow);
#ifdef OMPL_MBM_HAVE_VAMP
    reportRow("vamp-rrtc", "VAMP", overall, vampRow);
#endif
    std::printf("\n\"samples\" is checked configurations for isSafe, barrier evaluations for\n"
                "the three CBF rows, and SIMD configuration lanes evaluated for VAMP. They measure\n"
                "sampling work, while wall time captures the very different per-sample costs.\n"
                "\"unsafe/audited\" evaluates exact primitive clearance at every state of the\n"
                "densified solution -- for each CBF row that is the rollout the planner recorded\n"
                "for each edge, replayed, so it is the motion that would actually be executed,\n"
                "sampled inside each step rather than only at its boundaries. Non-zero unsafe\n"
                "invalidates a row however fast it was.\n"
                "\"missed\" counts solution edges that were not on file and had to be\n"
                "re-derived; it must be zero, or the audited motion is a different trajectory\n"
                "from the one the planner found.\n");
    std::printf("\n");
    {
        const auto &e = ompl::demo::EnvelopeHoldFilter::aggregate();
        if (e.calls > 0)
        {
            const double n = static_cast<double>(e.calls);
            std::printf("\nenvelope hop: %zu filter calls, %.3f hold queries/call, "
                        "%.1f%% saturated, "
                        "%.1f%% longer than L1, past one step %.1f%% (L1 alone %.1f%%), "
                        "gated %.1f%%, empty-active %.1f%%, screen-escalated %.1f%%\n",
                        e.calls, e.queries / n, 100.0 * e.saturated / e.queries,
                        100.0 * e.longer / n, 100.0 * e.pastStep / n,
                        100.0 * e.pastStepL1 / n,
                        100.0 * e.gated / static_cast<double>(e.gated + e.calls),
                        e.queries > 0 ? 100.0 * e.emptyActive / static_cast<double>(e.queries)
                                      : 0.0,
                        100.0 * e.rescreened / n);
            // Which geometry the hop is paying for. "unbound" is not a bottleneck: the
            // certificate reached the end of its horizon, or of the screening clip,
            // with clearance to spare, so a longer look -- not more room -- is what
            // that call wanted.
            std::printf("envelope bottleneck: world %.1f%%, self-collision %.1f%%, "
                        "unbound %.1f%% (horizon or screen clip), blocked %.1f%%, "
                        "joint-limit clamped %.1f%%\n",
                        100.0 * e.worldBound / n, 100.0 * e.selfBound / n,
                        100.0 * e.unbound / n, 100.0 * e.blocked / n,
                        100.0 * e.limitBound / n);
            // Within the binding row, which of its three lower bounds was the answer.
            // Shares are of that family's own bound calls, so the two lines read
            // independently; ties go to the cheapest term, so a large L1 share means
            // the envelope terms are being computed and not used.
            // The third term is not the same certificate in the two families and must
            // not be printed under one name: a self pair has a contact normal and gets
            // the directional bound, a world row has none and gets the strictly weaker
            // isotropic one. See `HoldEngine::worldRow`.
            const auto terms = [](const char *label, const char *anchor,
                                  const std::size_t (&t)[4])
            {
                const double d = static_cast<double>(t[0] + t[1] + t[2] + t[3]);
                if (d <= 0.0)
                    return;
                std::printf("  %-16s L1 %.1f%%, local-L1 %.1f%%, speed-capped %.1f%%, "
                            "%s %.1f%% (of %.0f bound calls)\n",
                            label, 100.0 * t[0] / d, 100.0 * t[1] / d, 100.0 * t[2] / d,
                            anchor, 100.0 * t[3] / d, d);
            };
            if (e.certificateSeconds > 0.0 || e.innerSeconds > 0.0)
            {
                const double calls = n + static_cast<double>(e.gated);
                std::printf("envelope cost: wrapped QP %.3f us/call, certificate %.3f "
                            "us/call (%.1f%% on top), of which gate-rejected %.3f us/call; "
                            "%.3f us per query run\n",
                            1e6 * e.innerSeconds / calls, 1e6 * e.certificateSeconds / calls,
                            e.innerSeconds > 0.0
                                ? 100.0 * e.certificateSeconds / e.innerSeconds
                                : 0.0,
                            1e6 * e.gatedSeconds / calls,
                            e.queries > 0
                                ? 1e6 * (e.certificateSeconds - e.gatedSeconds) /
                                      static_cast<double>(e.queries)
                                : 0.0);
            }
            std::printf("envelope binding term:\n");
            terms("world rows", "isotropic-anchored", e.worldTerm);
            terms("self-pair rows", "normal-anchored", e.selfTerm);
            // The number that decides whether the local-lever term pays: world rows it
            // settled before any envelope was built, over rows that got that far.
            if (const std::size_t reached = holdtime::localReachedTally(); reached > 0)
                std::printf("local-lever screen: %zu of %zu expensive world rows settled "
                            "before the envelope (%.1f%%)\n",
                            holdtime::localPrunedTally(), reached,
                            100.0 * holdtime::localPrunedTally() /
                                static_cast<double>(reached));

            // The statistic that survives contact with the rollout, and the one to quote
            // instead of `mean span gain`: the span the rollout is handed, in controller
            // steps. A hop is floored at one step, so only mass moving out of the first
            // column can change anything the planner does.
            const auto histogram = [](const char *label, const std::size_t (&h)[5])
            {
                double d = 0.0;
                for (int i = 0; i < 5; ++i)
                    d += static_cast<double>(h[i]);
                if (d <= 0.0)
                    return;
                std::printf("  %-9s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%%\n", label,
                            1e2 * h[0] / d, 1e2 * h[1] / d, 1e2 * h[2] / d, 1e2 * h[3] / d,
                            1e2 * h[4] / d);
            };
            std::printf("span handed to the rollout, in controller steps "
                        "(%zu calls):\n", e.calls + e.gated);
            std::printf("  %-9s %9s %8s %8s %8s %8s\n", "", "<1", "1-2", "2-4", "4-8", "8+");
            histogram("L1 only", e.l1Steps);
            histogram("envelope", e.envSteps);
        }
    }
    if (holdtime::holdTimingEnabled())
    {
        const holdtime::HoldTiming &t = holdtime::holdTiming();
        std::printf("\nhold certificate: %zu queries, %.3f s total, %.3f us/query, "
                    "%.2f self + %.2f world expensive rows/query\n",
                    t.calls, t.seconds,
                    t.calls > 0 ? 1e6 * t.seconds / static_cast<double>(t.calls) : 0.0,
                    t.calls > 0 ? static_cast<double>(t.selfRows) / static_cast<double>(t.calls)
                                : 0.0,
                    t.calls > 0 ? static_cast<double>(t.worldRows) / static_cast<double>(t.calls)
                                : 0.0);
    }
    ompl::cbf::Profiler::instance().report(stdout);
    ompl::cbf::FilterStats::instance().report(stdout);
    return 0;
}
