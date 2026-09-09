// Planner-level A/B: does the tighter self-pair certificate make RRTConnect faster?
//
// Both arms are CertifiedRegionRollout's walk. They differ in ONE thing: how the
// certified step along the ray is computed for the 303 self-collision rows.
//
//   arm A  region     : safeScale over all 343 rows (the L1 lever-arm polytope)
//   arm B  region+hold: world rows unchanged, self rows from the hold-time
//                       certificate max(level2, normal-anchored, L1)
//
// Same scene, same problems, same planner seed. Reports whole-solve wall time.
//
//   ./planner_ab <scene.grid> [problems] [seconds] [range] [seed]

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/CertifiedRegionRollout.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/robots/UR5.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>
#include <ompl/util/Time.h>
#include <ompl/util/Console.h>

#include "HoldTimeCertificate.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using UR5 = ompl::robots::UR5;
using Barrier = ompl::cbf::ClearanceBarrier;
using Filter = ompl::cbf::CBFControlFilter;
using Space = ompl::cbf::FilteredStateSpace;
using Configuration = UR5::Configuration;

// ---------------------------------------------------------------------------
// arm B: CertifiedRegionRollout's walk with the hold-time self-pair certificate
// ---------------------------------------------------------------------------
class HoldTimeRegionRollout
{
public:
    struct Stats
    {
        std::size_t edges{0}, evaluations{0}, pairsEvaluated{0};
        std::size_t completed{0}, blocked{0}, floored{0}, budgeted{0};
        std::size_t worldEvaluated{0};
        std::size_t selfBinds{0};      ///< evaluations where a self row, not a world row, set the step
        std::size_t selfBindsHelped{0};///< ...and the hold certificate beat the L1 one there
        double gainWhenBinding{0.0};   ///< sum of holdSelf/l1Self over those
        double travel{0.0};
    };

    HoldTimeRegionRollout(const Barrier &barrier, const UR5 &robot, bool worldHold = false)
      : barrier_(barrier), robot_(robot), worldHold_(worldHold)
    {
    }

    // World rows only: the L1 polytope on the ray, exactly as safeScale does,
    // restricted to the 40 sphere constraints.
    static double worldScale(const Barrier::CertifiedRegion &region, const Barrier::Values &travel)
    {
        if (!region.valid)
            return 0.0;
        // Mirrors safeScale exactly, restricted to the 40 world rows: a constraint no
        // moving joint can affect is SKIPPED, not treated as blocking. Returning 0 for a
        // zero-slack row before checking travel makes arm B block where arm A proceeds.
        double best = std::numeric_limits<double>::infinity();
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(UR5::nSpheres); ++i)
        {
            const double rate = travel[i];
            if (rate <= 0.0)
                continue;
            if (region.slack[i] < rate * best)
                best = region.slack[i] / rate;
        }
        return best;
    }

    bool plan(const Configuration &from, const Configuration &to, double fraction,
              Space::Rollout &out) const
    {
        constexpr double shrink = 1.0 - 1e-9;
        const double share = std::clamp(fraction, 0.0, 1.0);
        const Configuration target = from + share * (to - from);

        out = Space::Rollout();
        out.end = from;
        out.waypoints.push_back(from);
        ++stats_.edges;

        if (share <= 0.0 || (target - from).squaredNorm() <= 0.0)
        {
            ++stats_.completed;
            out.reachedTarget = share >= 1.0;
            return true;
        }

        const Configuration delta = target - from;
        const double dirNorm = delta.norm();
        const Configuration u = delta / dirNorm;
        const Configuration au = u.cwiseAbs();
        const Barrier::Values travel = Barrier::travelBound(delta);
        travelUnit_ = Barrier::travelBound(u);

        Configuration q = from;
        double covered = 0.0;
        while (covered < 1.0)
        {
            if (out.steps >= 40)
            {
                out.callBudgetReached = true;
                ++stats_.budgeted;
                break;
            }
            const Barrier::CertifiedRegion region = barrier_.certifiedRegion(q);

            const double left = 1.0 - covered;
            const double horizonArc = left * dirNorm;

            const UR5::Kinematics kin = robot_.kinematics(q);
            UR5::SphereCenters centers;
            UR5::sphereCenters(kin, centers);
            cache_.build(kin, u, horizonArc);

            // A certificate clipped at its own horizon means "the whole remainder is
            // certified", which must read as non-binding (infinite scale) -- not as a
            // step that stops there, or `reachedTarget` can never fire.
            auto asScale = [&](double arc)
            {
                return arc >= horizonArc * (1.0 - 1e-9)
                           ? std::numeric_limits<double>::infinity()
                           : arc / dirNorm;
            };

            long pairs = 0;
            const double selfArc = holdtime::holdSelfScale(kin, centers, u, au, horizonArc,
                                                           cache_, false, &pairs,
                                                           barrier_.selfMargin(), true);
            stats_.pairsEvaluated += static_cast<std::size_t>(pairs);
            const double selfDir = asScale(selfArc);

            double worldDir;
            if (worldHold_)
            {
                long wn = 0;
                worldDir = asScale(holdtime::holdWorldScale(kin, centers, region, travelUnit_,
                                                            horizonArc, cache_, &wn));
                stats_.worldEvaluated += static_cast<std::size_t>(wn);
            }
            else
            {
                worldDir = worldScale(region, travel);
            }

            if (selfDir < worldDir)
            {
                ++stats_.selfBinds;
                const double l1Self = holdtime::repoSelfScale(centers, au, horizonArc,
                                                              barrier_.selfMargin());
                if (selfArc > l1Self * (1.0 + 1e-9)) ++stats_.selfBindsHelped;
                if (l1Self > 0.0) stats_.gainWhenBinding += selfArc / l1Self;
            }

            const double dirScale = std::min(worldDir, selfDir);
            const double scale = dirScale / left;
            if (!(scale > 0.0))
            {
                out.blocked = 1;
                ++stats_.blocked;
                ++stats_.evaluations;
                break;
            }
            ++out.steps;
            ++stats_.evaluations;

            const double step = std::min(scale * shrink, 1.0);
            const Configuration remaining = target - q;
            const Configuration landing = q + step * remaining;
            const double advance = (landing - q).norm();
            if (step < 1.0 && advance < 0.01)
            {
                out.stalled = true;
                ++stats_.floored;
                break;
            }
            out.travel += advance;
            out.coarse += advance > 0.0 ? 1u : 0u;
            q = landing;
            out.waypoints.push_back(q);
            covered += step * left;
            if (step >= 1.0)
            {
                ++stats_.completed;
                break;
            }
        }

        out.end = q;
        out.fraction = share * covered;
        out.reachedTarget = covered >= 1.0 && share >= 1.0;
        stats_.travel += out.travel;
        return true;
    }

    Space::RolloutPlanner planner() const
    {
        return [this](const Configuration &f, const Configuration &t, double fr,
                      Space::Rollout &o) { return plan(f, t, fr, o); };
    }

    const Stats &stats() const { return stats_; }
    void reset() const { stats_ = Stats(); }

private:
    const Barrier &barrier_;
    const UR5 &robot_;
    bool worldHold_{false};
    mutable Barrier::Values travelUnit_;
    mutable holdtime::HoldCache cache_;
    mutable Stats stats_;
};

// ---------------------------------------------------------------------------

struct Problem
{
    Configuration start, goal;
};

struct Outcome
{
    double seconds{0.0};
    bool solved{false};
    std::size_t vertices{0};
    double length{0.0};
};

static Outcome solveOne(const Barrier &barrier, const Filter &filter, const Problem &prob,
                        double seconds, double range, const Space::RolloutPlanner *rollout)
{
    auto space = std::make_shared<Space>(filter, 0.05, UR5::velocityLimits());
    ob::RealVectorBounds bounds(Space::dimension);
    for (int j = 0; j < Space::dimension; ++j)
    {
        bounds.setLow(j, UR5::lowerBounds()[j]);
        bounds.setHigh(j, UR5::upperBounds()[j]);
    }
    space->setBounds(bounds);
    space->setMaxStepScale(1e9);
    Space::EarlyTermination et;
    et.enabled = true;
    space->setEarlyTermination(et);
    if (rollout != nullptr)
        space->setRolloutPlanner(*rollout);

    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
    si->setMotionValidator(std::make_shared<ompl::cbf::FilteredMotionValidator>(si));
    si->setup();

    ob::ScopedState<ob::RealVectorStateSpace> s(space), g(space);
    for (int j = 0; j < Space::dimension; ++j)
    {
        s->values[j] = prob.start[j];
        g->values[j] = prob.goal[j];
    }
    auto pdef = std::make_shared<ob::ProblemDefinition>(si);
    pdef->setStartAndGoalStates(s, g, 0.35);

    auto planner = std::make_shared<og::RRTConnect>(si);
    planner->setRange(range);
    planner->setRetainPartialSteering(true);
    planner->setProblemDefinition(pdef);
    planner->setup();

    const ompl::time::point begin = ompl::time::now();
    const ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(seconds));
    Outcome o;
    o.seconds = ompl::time::seconds(ompl::time::now() - begin);
    o.solved = (status == ob::PlannerStatus::EXACT_SOLUTION);
    ob::PlannerData data(si);
    planner->getPlannerData(data);
    o.vertices = data.numVertices();
    if (o.solved)
        o.length = pdef->getSolutionPath()->length();
    return o;
}

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <scene.grid> [problems] [seconds] [range] [seed]\n", argv[0]);
        return 1;
    }
    const int nProblems = argc > 2 ? std::atoi(argv[2]) : 40;
    const double seconds = argc > 3 ? std::atof(argv[3]) : 5.0;
    const double range = argc > 4 ? std::atof(argv[4]) : 1.5;
    const std::uint_fast32_t seed = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 1;

    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    ompl::RNG::setSeed(seed);
    const ompl::sdf::GridSDF field = ompl::sdf::GridSDF::load(argv[1]);
    const UR5 robot;
    const Barrier barrier(robot, field, Barrier::defaultMargin);
    Filter::Parameters fp;
    fp.kappa = 8.0;
    fp.maxSpeed = UR5::velocityLimits();
    const Filter filter(barrier, fp);

    // Problems: safe start and goal, far enough apart to need a search.
    std::mt19937 rng(seed);
    std::vector<Problem> problems;
    while (static_cast<int>(problems.size()) < nProblems)
    {
        Problem p;
        for (int j = 0; j < Space::dimension; ++j)
        {
            std::uniform_real_distribution<double> d(UR5::lowerBounds()[j], UR5::upperBounds()[j]);
            p.start[j] = d(rng);
            p.goal[j] = d(rng);
        }
        if (!barrier.isSafe(p.start) || !barrier.isSafe(p.goal))
            continue;
        if ((p.goal - p.start).norm() < 2.0)
            continue;
        // A problem whose straight edge is already free is solved by one extension and
        // measures nothing. Keep the ones where the direct rollout falls well short.
        {
            const ompl::cbf::CertifiedRegionRollout probe(barrier);
            Space::Rollout r;
            probe.plan(p.start, p.goal, 1.0, r);
            const double before = (p.goal - p.start).norm();
            const double after = (p.goal - r.end).norm();
            if (before <= 0.0 || (before - after) / before > 0.5)
                continue;
        }
        problems.push_back(p);
    }

    const ompl::cbf::CertifiedRegionRollout regionRollout(barrier);
    const bool worldHold = std::getenv("WORLD_HOLD") != nullptr;
    const HoldTimeRegionRollout holdRollout(barrier, robot, worldHold);
    std::printf("arm B: self rows = hold-time, world rows = %s\n", worldHold ? "hold-time" : "L1");
    const Space::RolloutPlanner regionPlanner = regionRollout.planner();
    const Space::RolloutPlanner holdPlanner = holdRollout.planner();

    std::vector<double> tRegion, tHold, vRegion, vHold, lRegion, lHold;
    int solvedRegion = 0, solvedHold = 0;
    for (int i = 0; i < nProblems; ++i)
    {
        ompl::RNG::setSeed(seed + static_cast<std::uint_fast32_t>(i + 1));
        const Outcome a = solveOne(barrier, filter, problems[i], seconds, range, &regionPlanner);
        ompl::RNG::setSeed(seed + static_cast<std::uint_fast32_t>(i + 1));
        const Outcome b = solveOne(barrier, filter, problems[i], seconds, range, &holdPlanner);
        solvedRegion += a.solved ? 1 : 0;
        solvedHold += b.solved ? 1 : 0;
        if (a.solved && b.solved)
        {
            tRegion.push_back(a.seconds);
            tHold.push_back(b.seconds);
            vRegion.push_back(double(a.vertices));
            vHold.push_back(double(b.vertices));
            lRegion.push_back(a.length);
            lHold.push_back(b.length);
        }
        std::printf("  [%2d] region %6.3fs %s (%zu v)   hold %6.3fs %s (%zu v)\n", i, a.seconds,
                    a.solved ? "ok  " : "FAIL", a.vertices, b.seconds, b.solved ? "ok  " : "FAIL",
                    b.vertices);
        std::fflush(stdout);
    }

    std::printf("\n=== planner A/B, %s, %d problems, %.1fs limit, range %.2f ===\n", argv[1],
                nProblems, seconds, range);
    std::printf("  solved            region %d/%d      hold %d/%d\n", solvedRegion, nProblems,
                solvedHold, nProblems);
    std::printf("  both solved       %zu problems\n", tRegion.size());
    if (!tRegion.empty())
    {
        double sr = 0.0, sh = 0.0;
        for (std::size_t i = 0; i < tRegion.size(); ++i) { sr += tRegion[i]; sh += tHold[i]; }
        std::printf("  solve time        region %.4fs median, %.4fs mean\n", median(tRegion),
                    sr / tRegion.size());
        std::printf("                    hold   %.4fs median, %.4fs mean\n", median(tHold),
                    sh / tHold.size());
        std::printf("  wall-time ratio   %.3fx median-of-totals, %.3fx on means\n",
                    median(tHold) / median(tRegion), sh / sr);
        std::printf("  vertices          region %.0f median   hold %.0f median\n", median(vRegion),
                    median(vHold));
        std::printf("  path length       region %.3f median   hold %.3f median\n", median(lRegion),
                    median(lHold));
    }
    const auto &ra = regionRollout.statistics();
    const auto &hb = holdRollout.stats();
    std::printf("\n  rollout work (summed over all solves)\n");
    std::printf("    region  edges %zu  evaluations %zu  (%.2f ev/edge)  travel %.1f rad\n",
                ra.edges, ra.evaluations, ra.edges ? double(ra.evaluations) / ra.edges : 0.0,
                ra.travel);
    std::printf("    hold    edges %zu  evaluations %zu  (%.2f ev/edge)  travel %.1f rad"
                "  pairs/eval %.1f\n",
                hb.edges, hb.evaluations, hb.edges ? double(hb.evaluations) / hb.edges : 0.0,
                hb.travel, hb.evaluations ? double(hb.pairsEvaluated) / hb.evaluations : 0.0);
    std::printf("\n  WHY: is the self-collision half even the binding constraint?\n");
    std::printf("    self row set the step in %zu of %zu evaluations  (%.1f%%)\n",
                hb.selfBinds, hb.evaluations,
                hb.evaluations ? 100.0 * hb.selfBinds / hb.evaluations : 0.0);
    std::printf("    ...and beat the L1 bound there in %zu  (%.1f%% of binding cases)\n",
                hb.selfBindsHelped, hb.selfBinds ? 100.0 * hb.selfBindsHelped / hb.selfBinds : 0.0);
    std::printf("    mean hold/L1 ratio when self binds  %.3fx\n",
                hb.selfBinds ? hb.gainWhenBinding / hb.selfBinds : 0.0);
    if (ra.edges && hb.edges)
        std::printf("    evaluations per edge ratio  %.3fx\n",
                    (double(hb.evaluations) / hb.edges) / (double(ra.evaluations) / ra.edges));
    return 0;
}
