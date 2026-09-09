// demos/CBFSteerBenchmark.cpp
//
// Times the steering primitive on its own.
//
//     ./build/demos/demo_CBFSteerBenchmark <scene.grid> [edges] [seed] [range] [kappa]
//
// The planner-level A/B in results/picard_sliding_benchmark measures whole solves,
// where the two steering methods build different trees and the comparison carries the
// search's variance with it. This does not: it draws the same edges for every method,
// steers each edge with each, and reports what that cost. What it cannot tell you is
// whether cheaper edges make a faster planner -- a rollout that ends somewhere else is
// a different tree -- so it is a kernel measurement to profile against, not a verdict.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/ParallelPicardRollout.h>
#include <ompl/cbf/Profiler.h>
#include <ompl/robots/UR5.h>
#include <ompl/sdf/GridSDF.h>

namespace
{
    using UR5 = ompl::robots::UR5;
    using Barrier = ompl::cbf::ClearanceBarrier;
    using Filter = ompl::cbf::CBFControlFilter;
    using Space = ompl::cbf::FilteredStateSpace;

    struct Totals
    {
        double seconds{0.0};
        std::size_t calls{0};
        std::size_t waypoints{0};
        double travel{0.0};
        double fraction{0.0};
        std::size_t reached{0};
        std::size_t blocked{0};
        std::size_t stalled{0};
        std::size_t tiny{0};
        std::size_t stillborn{0};  ///< produced no motion at all: a wasted sample
        /// How much of the gap to the requested target the extension actually closed.
        /// This is the sample-efficiency question: an extension that ends no nearer the
        /// target than it started gave the tree a vertex and no reach.
        double progress{0.0};
        std::size_t stuck{0};      ///< closed 5% of the gap or less
        std::size_t backwards{0};  ///< ended further from the target than it started
    };

    void report(const char *name, const Totals &t, std::size_t edges, const Totals &baseline)
    {
        std::printf("%-24s %8.3f us/edge  %7.2f calls/edge  %6.3f us/call  %6.3f rad  "
                    "%5.1f%% reached",
                    name, 1e6 * t.seconds / edges, double(t.calls) / edges,
                    t.calls > 0 ? 1e6 * t.seconds / t.calls : 0.0, t.travel / edges,
                    100.0 * t.reached / edges);
        if (baseline.seconds > 0.0)
            std::printf("   %5.2fx time  %5.2fx calls", t.seconds / baseline.seconds,
                        double(t.calls) / double(baseline.calls));
        std::printf("\n");
        // Sample efficiency, which is a different question from cost: how much of the
        // edge the extension actually covered, and how the ones that stopped short
        // stopped. A stillborn extension is a wasted sample outright -- the planner gets
        // no new vertex from it however cheap it was.
        std::printf("%-24s   %6.1f%% of horizon covered, %zu stillborn, %zu blocked, "
                    "%zu stalled, %zu tiny control\n",
                    "", 100.0 * t.fraction / edges, t.stillborn, t.blocked, t.stalled, t.tiny);
        std::printf("%-24s   %6.1f%% of the target gap closed, %zu stuck (<=5%%), "
                    "%zu ended further away\n",
                    "", 100.0 * t.progress / edges, t.stuck, t.backwards);
    }
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <scene.grid> [edges] [seed] [range] [kappa] [mode]\n"
                    "  mode 0: starts drawn uniformly, directions drawn uniformly\n"
                    "  mode 1: starts with barrier clearance under 20 mm\n"
                    "  mode 2: those starts, aimed straight down the worst barrier's\n"
                    "          gradient -- into the obstacle, which is where a pointwise\n"
                    "          CBF-QP deadlocks if it is going to\n",
                    argv[0]);
        return 1;
    }
    const std::size_t edges = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 2000;
    const std::uint_fast32_t seed = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 1;
    const double range = argc > 4 ? std::atof(argv[4]) : 1.5;
    const double kappa = argc > 5 ? std::atof(argv[5]) : 19.8;
    // Uniform starts almost never put a sphere in contact, and deadlock is a contact
    // phenomenon: the goal directly behind an active constraint. An RRT's nodes pile up
    // there, so a uniform probe measures the wrong population.
    const int mode = argc > 6 ? std::atoi(argv[6]) : 0;

    const ompl::sdf::GridSDF field = ompl::sdf::GridSDF::load(argv[1]);
    const UR5 robot;
    const Barrier barrier(robot, field, Barrier::defaultMargin);

    Filter::Parameters fp;
    fp.kappa = kappa;
    fp.maxSpeed = UR5::velocityLimits();
    const Filter filter(barrier, fp);

    Filter::Parameters fixedFp = fp;
    fixedFp.certificates = false;
    const Filter fixedFilter(barrier, fixedFp);

    constexpr double stepSize = 0.05;
    auto space = std::make_shared<Space>(filter, stepSize, UR5::velocityLimits());
    ompl::base::RealVectorBounds bounds(Space::dimension);
    for (int j = 0; j < Space::dimension; ++j)
    {
        bounds.setLow(j, UR5::lowerBounds()[j]);
        bounds.setHigh(j, UR5::upperBounds()[j]);
    }
    space->setBounds(bounds);
    auto fixedSpace = std::make_shared<Space>(fixedFilter, stepSize, UR5::velocityLimits());
    fixedSpace->setBounds(bounds);

    // Edges from safe starts toward a target `range` radians away, which is the shape
    // an RRT extension has. Drawn once and reused by every method.
    std::mt19937 rng(seed);
    std::vector<std::pair<UR5::Configuration, UR5::Configuration>> problems;
    problems.reserve(edges);
    while (problems.size() < edges)
    {
        UR5::Configuration from;
        UR5::Configuration direction;
        for (int j = 0; j < Space::dimension; ++j)
        {
            from[j] = std::uniform_real_distribution<double>(UR5::lowerBounds()[j],
                                                             UR5::upperBounds()[j])(rng);
            direction[j] = std::normal_distribution<double>(0.0, 1.0)(rng);
        }
        if (!barrier.isSafe(from))
            continue;
        if (mode > 0 && barrier.values(from).minCoeff() > 0.020)
            continue;
        if (mode > 1)
        {
            // Straight down dh/dq of the tightest barrier: the direction the filter must
            // refuse outright. Anything that still makes progress here is going around.
            const Barrier::Evaluation evaluation = barrier.evaluate(from);
            direction = -evaluation.rows.row(static_cast<Eigen::Index>(evaluation.worst))
                             .transpose();
            if (direction.norm() < 1e-9)
                continue;
        }
        UR5::Configuration to = from + range * direction.normalized();
        for (int j = 0; j < Space::dimension; ++j)
            to[j] = std::clamp(to[j], UR5::lowerBounds()[j], UR5::upperBounds()[j]);
        if ((to - from).norm() < 1e-6)
            continue;
        problems.emplace_back(from, to);
    }

    const auto time = [&](const std::function<Space::Rollout(const UR5::Configuration &,
                                                             const UR5::Configuration &)> &steer)
    {
        Totals t;
        const auto begin = std::chrono::steady_clock::now();
        for (const auto &problem : problems)
        {
            const Space::Rollout r = steer(problem.first, problem.second);
            const double before = (problem.second - problem.first).norm();
            const double after = (problem.second - r.end).norm();
            const double closed = before > 0.0 ? (before - after) / before : 0.0;
            t.progress += closed;
            t.stuck += closed <= 0.05 ? 1u : 0u;
            t.backwards += closed < 0.0 ? 1u : 0u;
            t.calls += r.steps + r.blocked;
            t.waypoints += r.waypoints.size();
            t.travel += r.travel;
            t.fraction += r.fraction;
            t.reached += r.reachedTarget ? 1u : 0u;
            t.blocked += r.blocked;
            t.stalled += r.stalled ? 1u : 0u;
            t.tiny += r.tinyControl ? 1u : 0u;
            t.stillborn += r.waypoints.size() < 2 ? 1u : 0u;
        }
        t.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        return t;
    };

    // One untimed pass so page faults and the SDF's first touches are not in the numbers.
    time([&](const UR5::Configuration &a, const UR5::Configuration &b)
         { return fixedSpace->roll(a, b, 1.0); });
    time([&](const UR5::Configuration &a, const UR5::Configuration &b)
         { return space->roll(a, b, 1.0); });

    const Totals fixed =
        time([&](const UR5::Configuration &a, const UR5::Configuration &b)
             { return fixedSpace->roll(a, b, 1.0); });
    const Totals sequential =
        time([&](const UR5::Configuration &a, const UR5::Configuration &b)
             { return space->roll(a, b, 1.0); });

    std::printf("%zu edges of %.2f rad, kappa %.1f/s, step %.3f s, mode %d, grid %s\n",
                edges, range, kappa, stepSize, mode, argv[1]);
    report("fixed step (no cert)", fixed, edges, fixed);
    report("certificate", sequential, edges, fixed);


    struct Configuration
    {
        const char *name;
        unsigned int window;
        unsigned int iterations;
        bool adaptive;
        bool converge;
    };
    // The last two ask the sample-efficiency question rather than the cost one: a window
    // iterated to a fixed point is the trajectory the sequential rollout integrates, so
    // if speculation could reach an extension the rollout misses, it would show up here
    // as a higher reached share or a longer covered fraction.
    const Configuration configurations[] = {
        {"picard fixed grid", 8, 2, false, false},
        {"picard adaptive", 8, 2, true, false},
        {"picard converged w=8", 8, 4, false, true},
        {"picard converged w=16", 16, 6, false, true},
    };

    for (const Configuration &configuration : configurations)
    {
        ompl::cbf::ParallelPicardRollout::Parameters p;
        p.workers = 1;
        p.windowSteps = configuration.window;
        p.maxIterations = configuration.iterations;
        p.minimumSteps = configuration.adaptive ? 1u : configuration.window;
        ompl::cbf::ParallelPicardRollout picard(filter, p);
        const Totals t =
            time([&](const UR5::Configuration &a, const UR5::Configuration &b) -> Space::Rollout
                 {
                     Space::Rollout out;
                     if (picard.plan(a, b, 1.0, stepSize, UR5::velocityLimits(),
                                     space->maxStepScale(), space->reachTolerance(), out))
                         return out;
                     // What the state space does with a refused proposal, so the fallback
                     // is charged to the method that needed it.
                     Space::Rollout fallback = space->roll(a, b, 1.0);
                     fallback.steps += out.steps;
                     return fallback;
                 });
        report(configuration.name, t, edges, sequential);
        const auto &s = picard.statistics();
        std::printf("    %zu windows, %zu map + %zu verify calls, %zu accepted, "
                    "%zu fallbacks, %zu direct certificates\n",
                    s.windows, s.mapFilterCalls, s.verificationFilterCalls, s.accepted,
                    s.fallbacks, s.directCertificates);
    }
    ompl::cbf::Profiler::instance().report();
    ompl::cbf::FilterStats::instance().report();
    return 0;
}
