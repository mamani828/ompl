// demos/UR5PyBulletSceneDemo.cpp
//
// Runs CBF steering on a workspace baked outside OMPL.
//
// The scenes come from a PyBullet setup (ur5_experiments/), which exports two
// files per environment: a signed distance grid in GridSDF's node layout, and a
// problem file holding the start configuration, goal configurations and joint
// limits. Nothing here knows what PyBullet is -- it loads a grid and plans.
//
//     ./build/demos/demo_UR5PyBulletScene <scene.problem> [seconds] [out.path]
//         [trials] [maxStepScale] [checkResolution] [baselineAudit] [selfMargin]
//         [shortcutDelta] [guardBuffer] [range] [minProgressFraction] [kappa]
//         [trialSeedOffset] [picardIterations] [picardWindow] [picardWorkers]
//     ./build/demos/demo_UR5PyBulletScene <scene.problem> --probe   < points.txt
//
// `--probe` reads "x y z" triples on stdin and prints the field value and
// gradient at each, so the Python baker can check that the grid it wrote is the
// field this side reads. That check is the whole point of a shared byte layout:
// a sign flip or an axis-order mistake in the exporter is otherwise invisible
// until the planner drives the arm through a wall.
//
// The goals are *not* validated here. The exporter picks configurations that
// maximise the barrier value precisely because being collision-free is not
// enough -- with a 0.06 m margin over a sphere model that under-covers the
// links, a mesh-free configuration inside a shelf bay is routinely infeasible.
// A goal with h < 0 is unreachable by construction: the filter would have to
// violate the barrier to arrive.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/ParallelPicardRollout.h>
#include <ompl/cbf/Profiler.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/robots/UR5.h>

#include <ompl/sdf/GridSDF.h>
#include <ompl/util/Time.h>

#include "UR5SelfCollisionAudit.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;
using Barrier = ompl::cbf::ClearanceBarrier;
using Filter = ompl::cbf::CBFControlFilter;
using UR5 = ompl::robots::UR5;
namespace sdf = ompl::sdf;

namespace
{
    constexpr int dimension = static_cast<int>(UR5::nJoints);

    /// One goal from the problem file: a configuration plus what it was for.
    double median(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    struct Goal
    {
        int index{0};
        std::string label;
        UR5::Configuration configuration;
        double exportedBarrier{0.0};  ///< h the exporter measured, for cross-checking
        Eigen::Vector3d target{Eigen::Vector3d::Zero()};
    };

    /// The exported scene: everything needed to set up the planning problem.
    struct Scene
    {
        std::string name;
        std::string gridPath;
        double voxel{0.02};
        double margin{Barrier::defaultMargin};
        UR5::Configuration start{UR5::Configuration::Zero()};
        UR5::Configuration lower{UR5::Configuration::Zero()};
        UR5::Configuration upper{UR5::Configuration::Zero()};
        std::vector<Goal> goals;
    };

    /// Ordinary uniform sampling with a per-run seed, so the two planner rows start from
    /// the same random sequence without pre-validating or rejection-sampling targets.
    class SeededStateSampler : public ob::RealVectorStateSampler
    {
    public:
        SeededStateSampler(const ob::StateSpace *space, std::uint_fast32_t seed)
          : ob::RealVectorStateSampler(space)
        {
            rng_.setLocalSeed(seed);
        }
    };

    std::string directoryOf(const std::string &path)
    {
        const std::size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    }

    Scene readScene(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("cannot open problem file " + path);

        Scene scene;
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream fields(line);
            std::string key;
            fields >> key;

            if (key == "env")
                fields >> scene.name;
            else if (key == "grid")
            {
                std::string file;
                fields >> file;
                scene.gridPath = directoryOf(path) + "/" + file;
            }
            else if (key == "voxel")
                fields >> scene.voxel;
            else if (key == "margin")
                fields >> scene.margin;
            else if (key == "start")
                for (int j = 0; j < dimension; ++j)
                    fields >> scene.start[j];
            else if (key == "limits_lower")
                for (int j = 0; j < dimension; ++j)
                    fields >> scene.lower[j];
            else if (key == "limits_upper")
                for (int j = 0; j < dimension; ++j)
                    fields >> scene.upper[j];
            else if (key == "goal")
            {
                Goal goal;
                fields >> goal.index >> goal.label;
                for (int j = 0; j < dimension; ++j)
                    fields >> goal.configuration[j];
                fields >> goal.exportedBarrier;
                for (int j = 0; j < 3; ++j)
                    fields >> goal.target[j];
                scene.goals.push_back(goal);
            }
        }
        if (scene.gridPath.empty())
            throw std::runtime_error("problem file names no grid: " + path);
        return scene;
    }

    /// Print field values on stdin points, for the exporter to diff against.
    int probe(const sdf::GridSDF &field)
    {
        const Eigen::Vector3i dims = field.dimensions();
        std::printf("# dims %d %d %d\n", dims[0], dims[1], dims[2]);
        std::printf("# spacing %.9f %.9f %.9f\n", field.spacing()[0], field.spacing()[1],
                    field.spacing()[2]);
        std::printf("# bounds %.9f %.9f %.9f %.9f %.9f %.9f\n", field.bounds().min()[0],
                    field.bounds().min()[1], field.bounds().min()[2], field.bounds().max()[0],
                    field.bounds().max()[1], field.bounds().max()[2]);
        std::printf("# maxGradientNorm %.9f\n", field.maxGradientNorm());

        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        while (std::scanf("%lf %lf %lf", &x, &y, &z) == 3)
        {
            const sdf::ValueGradient vg = field.valueAndGradient(Eigen::Vector3d(x, y, z));
            std::printf("%.9f %.9f %.9f %.9f\n", vg.value, vg.gradient[0], vg.gradient[1],
                        vg.gradient[2]);
        }
        return 0;
    }

    using ompl::demo::worstSelfOverlap;

    /// Self-collision margin for every barrier built here. A large negative value is the
    /// A/B for the rows themselves: pair clearances become enormous, so none is screened
    /// in and none caps the certificate -- the behaviour before the arm was modelled
    /// against itself. The PyBullet replay audits the meshes either way, so this is what
    /// makes it possible to ask what the rows are worth.
    double selfMargin = Barrier::defaultSelfMargin;

    /// Anchor spacing for the rope shortcut, in radians, applied to both rows after they
    /// solve. Non-positive disables it for an A/B. The 0.65 default is the coarse setting
    /// that retained the path-quality gain without the quadratic cost of dense anchors.
    double shortcutDelta = 0.65;

    /// World-clearance buffer enforced by the filter above the audited margin.
    /// Negative selects ClearanceBarrier::interpolationBuffer(field); zero is an
    /// intentionally unsafe experimental A/B that exposes interpolation violations.
    double guardBuffer = -1.0;

    /// Minimum fraction of straight-line progress needed to retain a deflected extension.
    /// The clutter benchmark uses 0.10 to keep over 90% of uniform samples productive.
    double minProgressFraction = 0.10;

    /// Negative keeps CBFControlFilter's default; exposed for benchmark A/B runs.
    /// CBF decay rate in 1/s (`dh/dt >= -kappa h`); negative leaves the filter default.
    double filterKappa = -1.0;

    std::uint_fast32_t trialSeedOffset = 0;
    unsigned int picardIterations = 0;
    unsigned int picardWindow = 8;
    unsigned int picardWorkers = 0;
    bool retainPartialSteering = true;
    bool rolloutEarlyTermination = true;
    unsigned int rolloutCallBudget = 40;
    unsigned int rolloutStalledSteps = 3;

    struct Result
    {
        bool solved{false};
        double seconds{0.0};
        std::size_t waypoints{0};
        std::size_t auditedStates{0};
        std::size_t unsafeStates{0};
        double minBarrier{std::numeric_limits<double>::infinity()};
        double minSelfOverlap{std::numeric_limits<double>::infinity()};
        std::size_t selfColliding{0};
        double goalDistance{0.0};
        std::size_t vertices{0};
        std::size_t blocked{0};
        std::size_t filtered{0};
        std::size_t steps{0};
        std::size_t coarse{0};  ///< steps the filter certified past `stepSize`
        double travel{0.0};     ///< joint-space radians rolled, so `travel / steps` is the
                                ///< distance one filter call bought
        std::size_t rollouts{0};   ///< planner rollout attempts, before shortcutting
        std::size_t recorded{0};   ///< planner edges retained from those rollouts
        std::size_t abandoned{0};  ///< rollout extensions suppressed for insufficient progress
        std::size_t served{0};     ///< motion queries answered from staged/recorded rollouts
        std::size_t primarySamples{0};   ///< primary RRTConnect sample extensions observed
        std::size_t productiveSamples{0};  ///< primary samples that added at least one node
        std::size_t proposalAttempts{0};
        std::size_t proposalAccepted{0};
        std::size_t proposalFallbacks{0};
        std::size_t callBudgetTerminations{0};
        std::size_t stallTerminations{0};
        std::size_t tinyControlTerminations{0};
        std::size_t picardWindows{0};
        std::size_t picardConvergedWindows{0};
        std::size_t picardMapCertifiedWindows{0};
        std::size_t picardMapCalls{0};
        std::size_t picardVerificationCalls{0};
        std::size_t picardVerificationFailures{0};
        double picardSeconds{0.0};
        std::size_t misses{0};   ///< solution edges re-derived rather than replayed
        std::size_t evicted{0};  ///< ledger edges dropped for capacity; must stay zero

        // Shortcutting, reported apart from the columns above. In particular
        // `shortcutSeconds` is *not* folded into `seconds`: it is post-processing, and a
        // sweep that costs more than the solve it shortens is a thing worth seeing.
        double lengthBefore{0.0};       ///< executed arc length before shortcutting, radians
        double lengthAfter{0.0};        ///< and after
        double shortcutSeconds{0.0};
        std::size_t shortcutsAccepted{0};
        std::size_t shortcutRollouts{0};  ///< candidate pairs that cost a rollout (cbf row only)
        std::size_t shortcutScreened{0};  ///< candidate pairs refused on geometry, at no QP cost
        double maxArrivalGap{0.0};        ///< longest uncertified snap a shortcut introduced
    };

    /// The bar: ordinary `geometric::RRTConnect` with straight-line edges and the same
    /// SDF behind an ordinary state validity checker. Same start, same goal, same goal
    /// tolerance, same range -- the only difference is how an edge is decided.
    ///
    /// Audited the same way too, but note the asymmetry that makes the clearance column
    /// worth reading: this row's motion between waypoints is a straight line, so
    /// densifying it reconstructs the executed motion exactly, whereas its *validity*
    /// was only ever sampled at `longestValidSegmentFraction`. The CBF row is the other
    /// way round -- every state it emits was certified as it was produced.
    Result planCollisionChecked(const Scene &scene, const sdf::GridSDF &field, const Goal &goal,
                                double timeLimit, double range, double checkResolution,
                                double auditSpacing, std::uint_fast32_t samplerSeed,
                                std::vector<UR5::Configuration> *path = nullptr)
    {
        const UR5 robot;
        const Barrier audit(robot, field, scene.margin, selfMargin);
        auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
        ob::RealVectorBounds bounds(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, scene.lower[j]);
            bounds.setHigh(j, scene.upper[j]);
        }
        space->setBounds(bounds);
        space->setStateSamplerAllocator(
            [samplerSeed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededStateSampler>(stateSpace, samplerSeed);
            });
        // Straight-line edges are only ever *sampled* for validity, and OMPL's default
        // samples them 15x coarser than this row is audited at -- which is a discount on
        // the checking, not a faster planner. Matching the two is what makes the timing
        // column mean the same thing in both rows.
        if (checkResolution > 0.0)
            space->setLongestValidSegmentFraction(checkResolution / space->getMaximumExtent());

        auto si = std::make_shared<ob::SpaceInformation>(space);
        std::size_t checks = 0;
        si->setStateValidityChecker(
            [&audit, &checks](const ob::State *state)
            {
                ++checks;
                return audit.isSafe(ompl::cbf::FilteredStateSpace::configurationOf(state));
            });
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> start(space);
        ob::ScopedState<ob::RealVectorStateSpace> target(space);
        for (int j = 0; j < dimension; ++j)
        {
            start->values[j] = scene.start[j];
            target->values[j] = goal.configuration[j];
        }

        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(start, target, 0.35);

        std::size_t primarySamples = 0;
        std::size_t productiveSamples = 0;
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(range);
        planner->setSampleExtensionCallback(
            [&primarySamples, &productiveSamples](const ob::State *, const ob::State *, bool,
                                                  bool productive, bool, bool)
            {
                ++primarySamples;
                productiveSamples += productive ? 1u : 0u;
            });
        planner->setProblemDefinition(pdef);
        planner->setup();

        const ompl::time::point begin = ompl::time::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));

        Result result;
        result.seconds = ompl::time::seconds(ompl::time::now() - begin);
        result.solved = (status == ob::PlannerStatus::EXACT_SOLUTION);
        result.steps = checks;
        result.primarySamples = primarySamples;
        result.productiveSamples = productiveSamples;

        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();

        if (pdef->hasSolution())
        {
            auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
            result.waypoints = solution->getStateCount();

            // Stock RRT-Rope. This row's edges *are* straight lines, so OMPL's own
            // implementation is exactly right for it and the CBF row's variant would be
            // wrong -- but both rows are shortcut at the same `delta`, which is what makes
            // the length columns comparable. Timed apart from the solve.
            result.lengthBefore = solution->length();
            if (shortcutDelta > 0.0)
            {
                const ompl::time::point shortcutBegin = ompl::time::now();
                og::PathSimplifier simplifier(si);
                // The return is only "did anything change", which the length columns say
                // better; rope's own accept count is not exposed.
                simplifier.ropeShortcutPath(*solution, shortcutDelta);
                result.shortcutSeconds = ompl::time::seconds(ompl::time::now() - shortcutBegin);
            }
            result.lengthAfter = solution->length();

            // Straight-line edges densify to the motion that would actually be executed,
            // so this row -- unlike the CBF row, which can only be audited at the
            // resolution its rollout emitted -- can be audited as finely as one likes.
            // That is what makes `auditSpacing` worth exposing: the row's validity was
            // only ever *sampled*, and shrinking the audit below the sampling is the
            // only way to find out whether the unsampled interior was safe too.
            solution->interpolate(static_cast<unsigned int>(solution->length() / auditSpacing));
            result.auditedStates = solution->getStateCount();
            for (std::size_t i = 0; i < solution->getStateCount(); ++i)
            {
                const UR5::Configuration q =
                    ompl::cbf::FilteredStateSpace::configurationOf(solution->getState(i));
                const double h = audit.worstValue(q);
                result.minBarrier = std::min(result.minBarrier, h);
                if (h < 0.0)
                    ++result.unsafeStates;

                const double own = worstSelfOverlap(robot, q);
                result.minSelfOverlap = std::min(result.minSelfOverlap, own);
                if (own < 0.0)
                    ++result.selfColliding;

                // The densified straight-line motion *is* what this row would execute, so
                // it is the thing an external mesh checker should be handed.
                if (path != nullptr)
                    path->push_back(q);
            }
            const UR5::Configuration last = ompl::cbf::FilteredStateSpace::configurationOf(
                solution->getState(solution->getStateCount() - 1));
            result.goalDistance = (last - goal.configuration).norm();
        }
        return result;
    }

    Result plan(const Scene &scene, const sdf::GridSDF &field, const Goal &goal, double timeLimit,
                double stepSize, double range, double maxStepScale, std::uint_fast32_t samplerSeed,
                std::vector<UR5::Configuration> *path)
    {
        const UR5 robot;
        const Barrier audit(robot, field, scene.margin, selfMargin);
        // The filter guards a *larger* margin than the one audited: enforcing
        // h >= 0 on the interpolated field only delivers h >= -O(voxel), so the
        // buffer is what turns "the QP was satisfied" into "the audit passes".
        const Barrier guard = Barrier::guarding(robot, field, scene.margin, guardBuffer, selfMargin);

        Filter::Parameters parameters;
        if (filterKappa > 0.0)
            parameters.kappa = filterKappa;
        const Filter filter(guard, parameters);

        auto space = std::make_shared<ompl::cbf::FilteredStateSpace>(filter, stepSize,
                                                                    UR5::velocityLimits());
        ob::RealVectorBounds bounds(dimension);
        for (int j = 0; j < dimension; ++j)
        {
            bounds.setLow(j, scene.lower[j]);
            bounds.setHigh(j, scene.upper[j]);
        }
        space->setBounds(bounds);
        space->setStateSamplerAllocator(
            [samplerSeed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededStateSampler>(stateSpace, samplerSeed);
            });
        // Above 1 the rollout runs each control as far as the filter certifies it, which
        // in open space is the whole extension; at 1 it steps at `stepSize` regardless.
        if (maxStepScale > 0.0)
            space->setMaxStepScale(maxStepScale);
        if (minProgressFraction >= 0.0)
            space->setMinProgressFraction(minProgressFraction);
        ompl::cbf::FilteredStateSpace::EarlyTermination earlyTermination;
        earlyTermination.enabled = rolloutEarlyTermination;
        earlyTermination.maxFilterCalls = rolloutCallBudget;
        earlyTermination.stalledSteps = rolloutStalledSteps;
        space->setEarlyTermination(earlyTermination);

        std::unique_ptr<ompl::cbf::ParallelPicardRollout> picard;
        if (picardIterations > 0)
        {
            ompl::cbf::ParallelPicardRollout::Parameters picardParameters;
            picardParameters.maxIterations = picardIterations;
            picardParameters.windowSteps = picardWindow;
            picardParameters.workers = picardWorkers;
            picard = std::make_unique<ompl::cbf::ParallelPicardRollout>(filter,
                                                                        picardParameters);
            space->setRolloutPlanner(
                [&picard, space](const UR5::Configuration &from,
                                 const UR5::Configuration &to, double fraction,
                                 ompl::cbf::FilteredStateSpace::Rollout &rollout)
                {
                    return picard->plan(from, to, fraction, space->stepSize(),
                                        space->maxSpeed(), space->maxStepScale(),
                                        space->reachTolerance(), rollout);
                });
        }
        auto si = std::make_shared<ob::SpaceInformation>(space);
        // The rollout certifies every step it emits, so there is nothing left for a
        // validity checker to do -- see README_CBF_USAGE.md on removing it.
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::FilteredMotionValidator>(si));
        si->setup();

        ob::ScopedState<ob::RealVectorStateSpace> start(space);
        ob::ScopedState<ob::RealVectorStateSpace> target(space);
        for (int j = 0; j < dimension; ++j)
        {
            start->values[j] = scene.start[j];
            target->values[j] = goal.configuration[j];
        }

        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        // A rollout cannot be asked to land exactly on a state beside an obstacle,
        // so the goal is a ball, matching the control-space demo's tolerance.
        pdef->setStartAndGoalStates(start, target, 0.35);

        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setRange(range);
        planner->setRetainPartialSteering(retainPartialSteering);
        std::size_t primarySamples = 0;
        std::size_t productiveSamples = 0;
        planner->setSampleExtensionCallback(
            [&primarySamples, &productiveSamples](const ob::State *, const ob::State *, bool,
                                                  bool productive, bool, bool)
            {
                ++primarySamples;
                productiveSamples += productive ? 1u : 0u;
            });
        planner->setProblemDefinition(pdef);
        planner->setup();

        // Exclude state-space setup/sanity work from planner statistics.
        space->resetStatistics();
        space->clearLedger();
        if (picard)
            picard->resetStatistics();

        const ompl::time::point begin = ompl::time::now();
        const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timeLimit));

        Result result;
        result.seconds = ompl::time::seconds(ompl::time::now() - begin);
        result.solved = (status == ob::PlannerStatus::EXACT_SOLUTION);

        const ompl::cbf::FilteredStateSpace::Statistics stats = space->statistics();
        result.steps = stats.steps;
        result.filtered = stats.filtered;
        result.blocked = stats.blocked;
        result.coarse = stats.coarse;
        result.travel = stats.travel;
        result.rollouts = stats.rollouts;
        result.recorded = stats.recorded;
        result.abandoned = stats.abandoned;
        result.served = stats.served;
        result.proposalAttempts = stats.proposalAttempts;
        result.proposalAccepted = stats.proposalAccepted;
        result.proposalFallbacks = stats.proposalFallbacks;
        result.callBudgetTerminations = stats.callBudgetTerminations;
        result.stallTerminations = stats.stallTerminations;
        result.tinyControlTerminations = stats.tinyControlTerminations;
        result.primarySamples = primarySamples;
        result.productiveSamples = productiveSamples;
        if (picard)
        {
            const auto &picardStats = picard->statistics();
            result.picardWindows = picardStats.windows;
            result.picardConvergedWindows = picardStats.convergedWindows;
            result.picardMapCertifiedWindows = picardStats.mapCertifiedWindows;
            result.picardMapCalls = picardStats.mapFilterCalls;
            result.picardVerificationCalls = picardStats.verificationFilterCalls;
            result.picardVerificationFailures = picardStats.verificationFailures;
            result.picardSeconds = picardStats.seconds;
        }

        ob::PlannerData data(si);
        planner->getPlannerData(data);
        result.vertices = data.numVertices();

        if (pdef->hasSolution())
        {
            auto solution = std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
            result.waypoints = solution->getStateCount();

            // RRT-Rope with the CBF rollout as the edge test. It leaves every edge of the
            // path in the ledger, so `misses` below is still the check that what gets
            // audited and written out is a replay -- and it is now checking the
            // shortcutter's sub-edge recording as well as the planner's.
            if (shortcutDelta > 0.0)
            {
                const ompl::time::point shortcutBegin = ompl::time::now();
                ompl::cbf::ShortcutReport shortcut;
                ompl::cbf::ropeShortcut(*solution, shortcutDelta, 0.1, &shortcut);
                result.shortcutSeconds = ompl::time::seconds(ompl::time::now() - shortcutBegin);
                result.lengthBefore = shortcut.lengthBefore;
                result.lengthAfter = shortcut.lengthAfter;
                result.shortcutsAccepted = shortcut.accepted;
                result.shortcutRollouts = shortcut.rollouts;
                result.shortcutScreened = shortcut.screened;
                result.maxArrivalGap = shortcut.maxArrivalGap;
                // The sweep rolls, so it is charged for what it rolled. Leaving it out
                // would make rad/call read as if the shortcut were free.
                const ompl::cbf::FilteredStateSpace::Statistics after = space->statistics();
                result.steps = after.steps;
                result.filtered = after.filtered;
                result.blocked = after.blocked;
                result.coarse = after.coarse;
                result.travel = after.travel;
            }
            // Densification writes a ledger edge per anchor, so the shortcut is the first
            // thing here able to push the ledger to its capacity. If it does, some edge
            // was dropped and `misses` below is about to find it.
            result.evicted = space->statistics().evicted;

            // Every edge replaced by the rollout the planner recorded for it, so this is
            // the motion that would actually be executed -- the only honest thing to
            // audit, and the only honest thing to write to the .path file below. The
            // resolution matches the rollout's own step travel, so the audit sees inside
            // each step rather than only its boundaries. `misses` must stay zero.
            const og::PathGeometric executed = ompl::cbf::executedPath(
                *solution, UR5::velocityLimits().maxCoeff() * stepSize, &result.misses);
            result.auditedStates = executed.getStateCount();
            // Measured on the executed motion rather than on the planner's polyline, which
            // is the only length this row can honestly quote: the straight line between
            // two of its nodes is not a motion the arm makes. Reported whether or not
            // shortcutting ran, so the two settings are comparable.
            result.lengthAfter = executed.length();
            if (shortcutDelta <= 0.0)
                result.lengthBefore = result.lengthAfter;
            for (std::size_t i = 0; i < executed.getStateCount(); ++i)
            {
                const UR5::Configuration q =
                    ompl::cbf::FilteredStateSpace::configurationOf(executed.getState(i));
                const double h = audit.worstValue(q);
                result.minBarrier = std::min(result.minBarrier, h);
                if (h < 0.0)
                    ++result.unsafeStates;

                const double own = worstSelfOverlap(robot, q);
                result.minSelfOverlap = std::min(result.minSelfOverlap, own);
                if (own < 0.0)
                    ++result.selfColliding;
                if (path != nullptr)
                    path->push_back(q);
            }
            const UR5::Configuration last = ompl::cbf::FilteredStateSpace::configurationOf(
                executed.getState(executed.getStateCount() - 1));
            result.goalDistance = (last - goal.configuration).norm();
        }
        return result;
    }
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <scene.problem> [seconds] [out.path] [trials] [maxStepScale]"
                    " [baselineCheckRadians] [baselineAuditRadians] [selfMargin]"
                    " [shortcutRadians] [guardBufferMeters] [rangeRadians]"
                    " [minProgressFraction] [kappa] [trialSeedOffset]"
                    " [picardIterations] [picardWindow] [picardWorkers]"
                    " [retainPartialSteering] [earlyTermination] [rolloutCallBudget]"
                    " [rolloutStalledSteps]\n",
                    argv[0]);
        std::printf("       %s <scene.problem> --probe   < points.txt\n", argv[0]);
        return 1;
    }

    const std::string problemPath = argv[1];
    const bool probeMode = (argc > 2 && std::string(argv[2]) == "--probe");
    const double timeLimit = (argc > 2 && !probeMode) ? std::atof(argv[2]) : 10.0;
    const std::string outPath = (argc > 3 && !probeMode) ? argv[3] : std::string();
    // Repeats for the timing columns only. A solve here is about a millisecond, so one
    // sample is noise; the reported path and audit come from the last trial.
    const int trials = (argc > 4 && !probeMode) ? std::max(1, std::atoi(argv[4])) : 5;
    // The certified-step A/B: 1 pins the rollout to `stepSize`, which is what it did
    // before the filter started reporting how long its answer stays good for.
    const double maxStepScale = (argc > 5 && !probeMode) ? std::atof(argv[5]) : -1.0;
    // Joint-space spacing the baseline checks its straight-line edges at, in radians.
    //
    // Negative now means "match the CBF rollout's step", not "leave OMPL's default". The
    // default is 0.01 of the state space's extent, which on this space is 0.154 rad --
    // six times coarser than the 0.025 rad the rollout advances per filter call. Comparing
    // the two rows at their own resolutions compares the resolutions: the baseline looks
    // fast because it is looking less often, and it is scored unsafe by an audit finer
    // than its own sampling. Tying them means both rows answer for the same spacing.
    const double checkResolutionArg = (argc > 6 && !probeMode) ? std::atof(argv[6]) : -1.0;
    // Joint-space spacing the baseline's *solution* is audited at, in radians. Negative
    // matches the CBF row's rollout step, which is the comparable setting; anything
    // finer asks whether the baseline's unsampled edge interiors were safe as well.
    const double baselineAudit = (argc > 7 && !probeMode) ? std::atof(argv[7]) : -1.0;
    if (argc > 8 && !probeMode)
        selfMargin = std::atof(argv[8]);
    // Rope anchor spacing, in radians. Non-positive leaves both rows unsimplified.
    if (argc > 9 && !probeMode)
        shortcutDelta = std::atof(argv[9]);
    if (argc > 10 && !probeMode)
        guardBuffer = std::atof(argv[10]);
    const double range = (argc > 11 && !probeMode) ? std::atof(argv[11]) : 1.5;
    if (argc > 12 && !probeMode)
        minProgressFraction = std::atof(argv[12]);
    if (argc > 13 && !probeMode)
        filterKappa = std::atof(argv[13]);
    if (argc > 14 && !probeMode)
        trialSeedOffset = static_cast<std::uint_fast32_t>(std::max(0, std::atoi(argv[14])));
    if (argc > 15 && !probeMode)
        picardIterations = static_cast<unsigned int>(std::max(0, std::atoi(argv[15])));
    if (argc > 16 && !probeMode)
        picardWindow = static_cast<unsigned int>(std::max(1, std::atoi(argv[16])));
    if (argc > 17 && !probeMode)
        picardWorkers = static_cast<unsigned int>(std::max(0, std::atoi(argv[17])));
    if (argc > 18 && !probeMode)
        retainPartialSteering = std::atoi(argv[18]) != 0;
    if (argc > 19 && !probeMode)
        rolloutEarlyTermination = std::atoi(argv[19]) != 0;
    if (argc > 20 && !probeMode)
        rolloutCallBudget = static_cast<unsigned int>(std::max(0, std::atoi(argv[20])));
    if (argc > 21 && !probeMode)
        rolloutStalledSteps = static_cast<unsigned int>(std::max(0, std::atoi(argv[21])));

    const Scene scene = readScene(problemPath);
    const sdf::GridSDF field = sdf::GridSDF::load(scene.gridPath);

    if (probeMode)
        return probe(field);

    const UR5 robot;
    const Barrier barrier(robot, field, scene.margin, selfMargin);
    if (guardBuffer < 0.0)
        guardBuffer = Barrier::interpolationBuffer(field);
    const double stepSize = 0.05;
    // What the CBF row's rollout emits, and so the finest spacing at which its executed
    // motion exists. The baseline is audited here too unless asked for something finer.
    const double auditSpacing = UR5::velocityLimits().maxCoeff() * stepSize;
    // The baseline checks at the same spacing the rollout steps at, unless overridden.
    const double checkResolution = checkResolutionArg > 0.0 ? checkResolutionArg : auditSpacing;

    std::printf("scene %s   grid %s\n", scene.name.c_str(), scene.gridPath.c_str());
    const Eigen::Vector3i dims = field.dimensions();
    std::printf("field %dx%dx%d nodes, voxel %.4f m, maxGradientNorm %.4f\n", dims[0], dims[1],
                dims[2], scene.voxel, field.maxGradientNorm());
    std::printf("margin %.4f, interpolation buffer %.4f, guarded margin %.4f\n", scene.margin,
                guardBuffer, scene.margin + guardBuffer);
    std::printf("CBF kappa %.3f /s, RRTConnect range %.3f rad, min progress %.2f\n",
                filterKappa > 0.0 ? filterKappa : Filter::Parameters().kappa, range,
                minProgressFraction >= 0.0 ? minProgressFraction : 0.25);
    std::printf("retain partial CBF steering: %s\n", retainPartialSteering ? "on" : "off");
    std::printf("rollout early termination: %s, %u-call budget, %u stalled steps\n",
                rolloutEarlyTermination ? "on" : "off", rolloutCallBudget,
                rolloutStalledSteps);
    if (picardIterations > 0)
    {
        const std::string workers =
            picardWorkers == 0 ? "automatic" : std::to_string(picardWorkers);
        std::printf("parallel Picard on: %u iterations, %u-step windows, %s workers\n",
                    picardIterations, picardWindow, workers.c_str());
    }
    // Confirm the two sides agree about the start before planning: a mismatch here
    // means the grid and the sphere model disagree about where the world is, and
    // every number after it would be measuring the wrong thing.
    const double startBarrier = barrier.worstValue(scene.start);
    std::printf("start barrier h = %+.4f m\n", startBarrier);
    if (startBarrier < 0.0)
    {
        std::printf("  start is already in violation; the filter cannot recover from this\n");
        return 2;
    }

    if (shortcutDelta > 0.0)
        std::printf("rope shortcut on, delta %.4f rad, both rows\n", shortcutDelta);

    std::printf("\n%-28s %-6s %7s %8s %6s %8s %8s %9s %7s %8s %6s %8s %9s %8s %8s %7s %6s\n",
                "goal", "row", "solved", "seconds", "wpts", "audited", "unsafe", "min h", "evals",
                "rad/call", "coarse", "selfcoll", "min self", "len0", "len1", "cut%", "cutms");

    std::vector<UR5::Configuration> combined;
    // The baseline's motion, written alongside so an external mesh checker can be run on
    // both rows. Without it the only path leaving this demo is the CBF one, and a
    // mesh-level finding about it cannot be attributed: the two rows share a sphere model
    // and a validity checker, so "does the plain planner do this too?" is the first
    // question any such finding raises.
    std::vector<UR5::Configuration> baseCombined;
    std::size_t solvedCount = 0;
    std::size_t unsafeTotal = 0;
    std::size_t selfTotal = 0;
    // Medians over `trials` repeats: a solve here takes about a millisecond, which is
    // the same order as the noise, so a single sample says nothing about which row is
    // faster. Only the timing is repeated -- the reported path is the last one.
    std::size_t baselineSolved = 0;
    std::vector<double> cbfTimes, baselineTimes;
    // Summed over the goals both rows solved, so the ratio at the end compares the two
    // rows on the same problems rather than on whichever each happened to manage.
    double cbfLength = 0.0, baselineLength = 0.0;
    double cbfLengthBefore = 0.0, baselineLengthBefore = 0.0;
    std::size_t bothSolved = 0;
    for (const Goal &goal : scene.goals)
    {
        const double exported = barrier.worstValue(goal.configuration);
        std::vector<UR5::Configuration> path;
        std::vector<UR5::Configuration> basePath;

        std::vector<double> cbfRun, baseRun;
        Result r, base;
        Result diagnostics;
        std::size_t baselinePrimarySamples = 0;
        std::size_t baselineProductiveSamples = 0;
        for (int trial = 0; trial < trials; ++trial)
        {
            path.clear();
            basePath.clear();
            const std::uint_fast32_t samplerSeed =
                0xcbf000u + trialSeedOffset +
                static_cast<std::uint_fast32_t>(goal.index * 10000 + trial);
            r = plan(scene, field, goal, timeLimit, stepSize, range, maxStepScale, samplerSeed,
                     &path);
            base = planCollisionChecked(scene, field, goal, timeLimit, range, checkResolution,
                                        baselineAudit > 0.0 ? baselineAudit : auditSpacing, samplerSeed,
                                        &basePath);
            cbfRun.push_back(r.seconds);
            baseRun.push_back(base.seconds);
            diagnostics.steps += r.steps;
            diagnostics.rollouts += r.rollouts;
            diagnostics.recorded += r.recorded;
            diagnostics.abandoned += r.abandoned;
            diagnostics.served += r.served;
            diagnostics.callBudgetTerminations += r.callBudgetTerminations;
            diagnostics.stallTerminations += r.stallTerminations;
            diagnostics.tinyControlTerminations += r.tinyControlTerminations;
            diagnostics.primarySamples += r.primarySamples;
            diagnostics.productiveSamples += r.productiveSamples;
            baselinePrimarySamples += base.primarySamples;
            baselineProductiveSamples += base.productiveSamples;
        }
        const double cbfMedian = median(cbfRun);
        const double baseMedian = median(baseRun);
        cbfTimes.push_back(cbfMedian);
        baselineTimes.push_back(baseMedian);

        solvedCount += r.solved ? 1 : 0;
        baselineSolved += base.solved ? 1 : 0;
        unsafeTotal += r.unsafeStates;
        selfTotal += r.selfColliding;
        if (r.solved && base.solved)
        {
            ++bothSolved;
            cbfLength += r.lengthAfter;
            baselineLength += base.lengthAfter;
            cbfLengthBefore += r.lengthBefore;
            baselineLengthBefore += base.lengthBefore;
        }

        // How much of the path the shortcut took off, as a percentage of what it started
        // from. This is the number the whole post-process exists to produce, and it is
        // computed the same way for both rows.
        const auto cutPercent = [](const Result &res)
        { return res.lengthBefore > 0.0 ? 1e2 * (1.0 - res.lengthAfter / res.lengthBefore) : 0.0; };

        std::printf("%-28s %-6s %7s %8.3f %6zu %8zu %8zu %+9.4f %7zu %8s %6s %8zu %+9.4f %8.3f "
                    "%8.3f %6.1f%% %6.1f\n",
                    goal.label.c_str(), "rrtc", base.solved ? "yes" : "no", baseMedian,
                    base.waypoints, base.auditedStates, base.unsafeStates,
                    base.solved ? base.minBarrier : 0.0, base.steps, "-", "-",
                    base.selfColliding, base.solved ? base.minSelfOverlap : 0.0, base.lengthBefore,
                    base.lengthAfter, cutPercent(base), 1e3 * base.shortcutSeconds);
        std::printf("%-28s %-6s %7s %8.3f %6zu %8zu %8zu %+9.4f %7zu %8.4f %5.0f%% %8zu %+9.4f "
                    "%8.3f %8.3f %6.1f%% %6.1f\n",
                    "", "cbf", r.solved ? "yes" : "no", cbfMedian, r.waypoints, r.auditedStates,
                    r.unsafeStates, r.solved ? r.minBarrier : 0.0, r.steps,
                    r.steps > 0 ? r.travel / r.steps : 0.0,
                    r.steps > 0 ? 1e2 * r.coarse / r.steps : 0.0, r.selfColliding,
                    r.solved ? r.minSelfOverlap : 0.0, r.lengthBefore, r.lengthAfter, cutPercent(r),
                    1e3 * r.shortcutSeconds);
        std::printf("    cbf planner: %zu recorded / %zu rollouts (%.1f%%), %zu abandoned, "
                    "%zu ledger-served, %zu filter calls\n",
                    diagnostics.recorded, diagnostics.rollouts,
                    diagnostics.rollouts > 0
                        ? 1e2 * static_cast<double>(diagnostics.recorded) / diagnostics.rollouts
                        : 0.0,
                    diagnostics.abandoned, diagnostics.served, diagnostics.steps);
        std::printf("    rollout early stops: %zu budget, %zu stalled, %zu tiny-control\n",
                    diagnostics.callBudgetTerminations, diagnostics.stallTerminations,
                    diagnostics.tinyControlTerminations);
        if (picardIterations > 0)
            std::printf("    parallel Picard: %zu/%zu proposals accepted, %zu fallbacks; "
                        "%zu/%zu windows converged (%zu map-certified), "
                        "%zu map + %zu verify filters, "
                        "%zu verification failures, %.3f ms total\n",
                        r.proposalAccepted, r.proposalAttempts, r.proposalFallbacks,
                        r.picardConvergedWindows, r.picardWindows,
                        r.picardMapCertifiedWindows, r.picardMapCalls,
                        r.picardVerificationCalls, r.picardVerificationFailures,
                        1e3 * r.picardSeconds);
        std::printf("    productive samples: rrtc %zu/%zu (%.1f%%), cbf %zu/%zu (%.1f%%)\n",
                    baselineProductiveSamples, baselinePrimarySamples,
                    baselinePrimarySamples > 0
                        ? 1e2 * static_cast<double>(baselineProductiveSamples) /
                              baselinePrimarySamples
                        : 0.0,
                    diagnostics.productiveSamples, diagnostics.primarySamples,
                    diagnostics.primarySamples > 0
                        ? 1e2 * static_cast<double>(diagnostics.productiveSamples) /
                              diagnostics.primarySamples
                        : 0.0);
        if (shortcutDelta > 0.0 && r.solved)
            std::printf("    cbf shortcut: %zu accepted from %zu rollouts, %zu pairs screened out, "
                        "max arrival gap %.4f rad\n",
                        r.shortcutsAccepted, r.shortcutRollouts, r.shortcutScreened,
                        r.maxArrivalGap);

        // Every solution edge should have been replayed from the rollout that made it.
        // If not, what was just audited -- and what is about to be written out -- is a
        // re-derived trajectory rather than the one the planner found.
        if (r.misses > 0)
            std::printf("    ! %zu solution edge(s) re-derived rather than replayed\n", r.misses);
        if (r.evicted > 0)
            std::printf("    ! %zu ledger edge(s) evicted for capacity; raise setLedgerCapacity "
                        "or coarsen the shortcut delta\n",
                        r.evicted);

        // The exporter measured h with its own copy of the sphere model; if that
        // disagrees with this one the two pipelines have drifted apart.
        if (std::abs(exported - goal.exportedBarrier) > 5e-3)
            std::printf("    ! barrier mismatch at goal: exporter %+.4f, here %+.4f\n",
                        goal.exportedBarrier, exported);

        if (!outPath.empty() && r.solved)
        {
            combined.push_back(scene.start);
            for (const UR5::Configuration &q : path)
                combined.push_back(q);
        }
        if (!outPath.empty() && base.solved)
        {
            baseCombined.push_back(scene.start);
            for (const UR5::Configuration &q : basePath)
                baseCombined.push_back(q);
        }
    }

    std::printf("\nsolved: rrtc %zu/%zu, cbf %zu/%zu   median ms over %d trials: rrtc %.3f, "
                "cbf %.3f (%.2fx)\n",
                baselineSolved, scene.goals.size(), solvedCount, scene.goals.size(), trials,
                1e3 * median(baselineTimes), 1e3 * median(cbfTimes),
                median(cbfTimes) > 0.0 ? median(baselineTimes) / median(cbfTimes) : 0.0);
    // The length comparison, over the goals both rows solved. Before shortcutting existed
    // this could only be quoted against an unsimplified baseline, which flattered the
    // filtered row: RRTConnect's raw output zigzags and rope takes most of that out. The
    // `after` ratio is the one that survives.
    if (bothSolved > 0)
        std::printf("path length over %zu goals both solved, radians: rrtc %.3f -> %.3f, "
                    "cbf %.3f -> %.3f   cbf/rrtc before %.2fx, after %.2fx\n",
                    bothSolved, baselineLengthBefore, baselineLength, cbfLengthBefore, cbfLength,
                    baselineLengthBefore > 0.0 ? cbfLengthBefore / baselineLengthBefore : 0.0,
                    baselineLength > 0.0 ? cbfLength / baselineLength : 0.0);
    std::printf("%zu audited states below the audited margin (cbf row)\n", unsafeTotal);
    if (selfTotal > 0)
        std::printf("%zu audited states self-collide. The barrier does model the arm against "
                    "itself now, so this is not a known gap any more -- it says a pair that "
                    "can collide is missing from UR5::selfPairs(). Re-run "
                    "scripts/generate_ur5_self_pairs.py with a larger --band\n",
                    selfTotal);

    if (!outPath.empty())
    {
        std::ofstream out(outPath);
        if (!out)
        {
            std::printf("cannot write %s\n", outPath.c_str());
            return 1;
        }
        out << "# joint-space path for " << scene.name << ", one configuration per line\n";
        for (const UR5::Configuration &q : combined)
        {
            for (int j = 0; j < dimension; ++j)
                out << (j ? " " : "") << q[j];
            out << "\n";
        }
        std::printf("wrote %s (%zu configurations)\n", outPath.c_str(), combined.size());

        const std::string basePathFile = outPath + ".rrtc";
        std::ofstream baseOut(basePathFile);
        if (baseOut)
        {
            baseOut << "# joint-space path for " << scene.name
                    << ", collision-checked RRTConnect, one configuration per line\n";
            for (const UR5::Configuration &q : baseCombined)
            {
                for (int j = 0; j < dimension; ++j)
                    baseOut << (j ? " " : "") << q[j];
                baseOut << "\n";
            }
            std::printf("wrote %s (%zu configurations)\n", basePathFile.c_str(),
                        baseCombined.size());
        }
    }
    if (ompl::cbf::profilingEnabled())
    {
        ompl::cbf::Profiler::instance().report(stdout);
        ompl::cbf::FilterStats::instance().report(stdout);
    }
    return solvedCount == scene.goals.size() ? 0 : 3;
}
