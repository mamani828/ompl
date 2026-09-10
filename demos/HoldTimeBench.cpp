// Microbenchmark and equivalence harness for the hold-time certificate.
//
// The MBM planner row `holdNew` spends its time in one query: given a
// configuration q, a unit joint-space direction u and a remaining arc length,
// how far along the ray is provably collision free. This program isolates that
// query, so its cost and its answer can both be measured without a planner in
// the way.
//
//   phase 1  replay the rollout with the FROZEN reference implementation and
//            record every (q, u, horizon) triple it asks about, so the query
//            distribution is the planner's own rather than something invented;
//   phase 2  run reference and optimized implementations over the recorded
//            triples, timing both and comparing every answer.
//
// The optimized answer must never exceed the reference one: larger means a
// collision row was skipped. Smaller by more than root-solver tolerance means
// the certificate was weakened. Both are reported.
//
//   ./demo_HoldTimeBench scenes.txt [perScene] [edges] [voxel] [margin]
//                        [buffer] [selfMargin] [seed] [reps]

#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/robots/UR5.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/Exception.h>

#include "HoldTimeCertificateRef.h"
#include "HoldTimeCertificate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Barrier = ompl::cbf::ClearanceBarrier;
using UR5 = ompl::robots::UR5;
using Configuration = UR5::Configuration;
namespace sdf = ompl::sdf;

namespace
{
    constexpr int dimension = 6;

    struct Obstacle
    {
        enum class Kind { Box, Cylinder };
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
            throw ompl::Exception("cannot open " + path);

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
                continue;
            else if (tag == "start" || tag == "goal")
            {
                Configuration q;
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

    // ---------------------------------------------------------------- reference

    /// Exactly the per-hop body of the pre-optimization HoldTimeRegionRollout::plan(),
    /// worldHold_ == true, returning the certified arc length.
    double referenceHop(const Barrier &barrier, const UR5 &robot, const Configuration &q,
                        const Configuration &u, double horizonArc,
                        holdtime_ref::HoldCache &cache, long *selfRows, long *worldRows)
    {
        const Configuration au = u.cwiseAbs();
        const Barrier::Values travelUnit = Barrier::travelBound(u);
        const Barrier::CertifiedRegion region = barrier.certifiedRegion(q);
        const UR5::Kinematics kin = robot.kinematics(q);
        UR5::SphereCenters centers;
        UR5::sphereCenters(kin, centers);
        cache.build(kin, u, horizonArc);

        long pairs = 0;
        const double selfArc = holdtime_ref::holdSelfScale(kin, centers, u, au, horizonArc, cache,
                                                           false, &pairs, barrier.selfMargin(),
                                                           true);
        long wn = 0;
        const double worldArc = region.valid
                                    ? holdtime_ref::holdWorldScale(kin, centers, region, travelUnit,
                                                                   horizonArc, cache, &wn)
                                    : 0.0;
        if (selfRows != nullptr)
            *selfRows += pairs;
        if (worldRows != nullptr)
            *worldRows += wn;
        return std::min(selfArc, worldArc);
    }

    struct Query
    {
        Configuration q;
        Configuration u;
        double horizon{0.0};
    };

    /// Replay the rollout walk to harvest the query distribution the planner produces.
    void collectQueries(const Barrier &barrier, const UR5 &robot, const Configuration &from,
                        const Configuration &to, std::vector<Query> &out)
    {
        constexpr double shrink = 1.0 - 1e-9;
        const Configuration delta = to - from;
        const double dirNorm = delta.norm();
        if (!(dirNorm > 0.0))
            return;
        const Configuration u = delta / dirNorm;
        holdtime_ref::HoldCache cache;

        Configuration q = from;
        double covered = 0.0;
        int steps = 0;
        while (covered < 1.0 && steps < 40)
        {
            const double left = 1.0 - covered;
            const double horizonArc = left * dirNorm;
            out.push_back({q, u, horizonArc});
            const double arc = referenceHop(barrier, robot, q, u, horizonArc, cache, nullptr,
                                            nullptr);
            const double dirScale = arc >= horizonArc * (1.0 - 1e-9)
                                        ? std::numeric_limits<double>::infinity()
                                        : arc / dirNorm;
            const double scale = dirScale / left;
            if (!(scale > 0.0))
                break;
            ++steps;
            const double step = std::min(scale * shrink, 1.0);
            const Configuration remaining = to - q;
            const Configuration landing = q + step * remaining;
            const double advance = (landing - q).norm();
            if (step < 1.0 && advance < 0.01)
                break;
            q = landing;
            covered += step * left;
            if (step >= 1.0)
                break;
        }
    }

    double now()
    {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s scenes.txt [perScene] [edges] [voxel] [margin] [buffer] "
                    "[selfMargin] [seed] [reps]\n",
                    argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    const int perScene = argc > 2 ? std::atoi(argv[2]) : 3;
    const int edges = argc > 3 ? std::atoi(argv[3]) : 40;
    const double voxel = argc > 4 ? std::atof(argv[4]) : 0.03;
    const double margin = argc > 5 ? std::atof(argv[5]) : 0.004;
    const double buffer = argc > 6 ? std::atof(argv[6]) : 0.005;
    const double selfMargin = argc > 7 ? std::atof(argv[7]) : 0.0;
    const unsigned long seed = argc > 8 ? std::strtoul(argv[8], nullptr, 10) : 1UL;
    const int reps = argc > 9 ? std::atoi(argv[9]) : 5;

    const std::vector<Problem> problems = readProblems(path);
    const UR5 robot;

    std::map<std::string, int> seen;
    std::vector<Problem> chosen;
    for (const Problem &p : problems)
        if (seen[p.scene]++ < perScene)
            chosen.push_back(p);

    double refSeconds = 0.0, optSeconds = 0.0;
    double geoSeconds = 0.0, regionSeconds = 0.0, fkSeconds = 0.0, buildSeconds = 0.0;
    double matvecSeconds = 0.0, screenSeconds = 0.0, legacyRootSeconds = 0.0;
    double sinkTotal = 0.0;
    std::size_t queries = 0;
    long refSelfRows = 0, refWorldRows = 0;
    double refSum = 0.0, optSum = 0.0;
    double worstAbove = 0.0, worstBelow = 0.0, worstRel = 0.0;
    std::size_t mismatches = 0, unsafeLarger = 0;
    double legacyWorst = 0.0;
    std::size_t legacyMismatches = 0;
    double reuseWorst = 0.0;
    std::size_t reuseMismatches = 0;
    std::vector<double> perQueryRefUs, perQueryOptUs;

    holdtime::resetCounters();

    for (const Problem &problem : chosen)
    {
        const sdf::GridSDF field(problem.field(), UR5::reachableBounds(), voxel);
        const Barrier guard = Barrier::guarding(robot, field, margin, buffer, selfMargin);

        std::mt19937 rng(static_cast<unsigned>(seed + problem.index * 7919u +
                                               std::hash<std::string>{}(problem.scene)));
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        auto sample = [&]
        {
            Configuration c;
            for (int j = 0; j < dimension; ++j)
                c[j] = UR5::lowerBounds()[j] +
                       unit(rng) * (UR5::upperBounds()[j] - UR5::lowerBounds()[j]);
            return c;
        };

        std::vector<Query> qs;
        // Anchor half the edges at the problem's own endpoints, which is where the
        // planner actually spends its rollouts, and half in free space.
        for (int e = 0; e < edges; ++e)
        {
            Configuration from = (e % 4 == 0) ? problem.start : (e % 4 == 1 ? problem.goal
                                                                            : sample());
            Configuration to = sample();
            Configuration d = to - from;
            const double n = d.norm();
            if (!(n > 0.0))
                continue;
            if (n > 2.0)
                to = from + d * (2.0 / n);
            collectQueries(guard, robot, from, to, qs);
        }
        if (qs.empty())
            continue;

        std::vector<double> refOut(qs.size()), optOut(qs.size()), legacyOut(qs.size());

        // warm up both, then time
        holdtime_ref::HoldCache refCache;
        holdtime::HoldEngine engine(guard);
        for (std::size_t i = 0; i < qs.size(); ++i)
        {
            refOut[i] = referenceHop(guard, robot, qs[i].q, qs[i].u, qs[i].horizon, refCache,
                                     nullptr, nullptr);
            optOut[i] = engine.holdScale(qs[i].q, qs[i].u, qs[i].horizon);
        }
        // Same engine, same screening, but the original bisection and no early exit
        // on the anchored term: this arm isolates the search from the solver, so a
        // difference here would mean a row was wrongly skipped rather than a root
        // resolved more finely.
        // The borrowed-geometry entry point must agree with the self-computing one:
        // it is the same query, differing only in who walked the chain.
        {
            Barrier::Evaluation eval;
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                guard.evaluate(qs[i].q, eval);
                const double a = engine.holdScale(qs[i].q, qs[i].u, qs[i].horizon);
                const double b = engine.holdScaleFrom(eval, qs[i].u, qs[i].horizon);
                const double d = std::abs(a - b);
                reuseWorst = std::max(reuseWorst, d);
                if (d > 0.0)
                    ++reuseMismatches;
            }
        }
        engine.setLegacyRoots(true);
        for (std::size_t i = 0; i < qs.size(); ++i)
            legacyOut[i] = engine.holdScale(qs[i].q, qs[i].u, qs[i].horizon);
        {
            const double t = now();
            for (int r = 0; r < reps; ++r)
                for (std::size_t i = 0; i < qs.size(); ++i)
                    optSum += engine.holdScale(qs[i].q, qs[i].u, qs[i].horizon);
            legacyRootSeconds += now() - t;
        }
        engine.setLegacyRoots(false);
        for (std::size_t i = 0; i < qs.size(); ++i)
        {
            const double d = std::abs(legacyOut[i] - refOut[i]);
            legacyWorst = std::max(legacyWorst, d);
            if (d > 0.0)
                ++legacyMismatches;
        }

        double t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
                refSum += referenceHop(guard, robot, qs[i].q, qs[i].u, qs[i].horizon, refCache,
                                       &refSelfRows, &refWorldRows);
        refSeconds += now() - t0;

        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
                optSum += engine.holdScale(qs[i].q, qs[i].u, qs[i].horizon);
        optSeconds += now() - t0;

        // Component floors, so the report says what is left to win rather than only
        // what has been won.
        Barrier::WorldRegion geo;
        double sink = 0.0;
        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                guard.worldRegion(qs[i].q, geo);
                sink += geo.slack[0];
            }
        geoSeconds += now() - t0;

        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
                sink += guard.certifiedRegion(qs[i].q).slack[0];
        regionSeconds += now() - t0;

        UR5::Kinematics kin;
        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                robot.kinematics(qs[i].q, kin);
                sink += kin.jointOrigin[5].x();
            }
        fkSeconds += now() - t0;

        Eigen::Matrix<double, UR5::nSelfPairs, 1> speeds;
        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                speeds.noalias() = UR5::selfPairLeverArms() * qs[i].u.cwiseAbs();
                sink += speeds[7];
            }
        matvecSeconds += now() - t0;

        // The 303-row screening pass on its own: gather, distance, slack, division.
        guard.worldRegion(qs[0].q, geo);
        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                speeds.noalias() = UR5::selfPairLeverArms() * qs[i].u.cwiseAbs();
                double m = qs[i].horizon;
                for (std::size_t pp = 0; pp < UR5::nSelfPairs; ++pp)
                {
                    const auto &pr = UR5::selfPairs()[pp];
                    const double d = (geo.centers.col(static_cast<Eigen::Index>(pr.a)) -
                                      geo.centers.col(static_cast<Eigen::Index>(pr.b)))
                                         .norm();
                    const double sl = d - UR5::selfPairRadii()[pp] - pr.margin;
                    const double l1 = sl / speeds[pp];
                    if (l1 < m) m = l1;
                }
                sink += m;
            }
        screenSeconds += now() - t0;

        holdtime::HoldCache probe;
        robot.kinematics(qs[0].q, kin);
        t0 = now();
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < qs.size(); ++i)
            {
                probe.build(kin, qs[i].u, qs[i].horizon, holdtime::SP.maxBaseFrame);
                sink += probe.cumV[0][5];
            }
        buildSeconds += now() - t0;
        sinkTotal += sink;

        for (std::size_t i = 0; i < qs.size(); ++i)
        {
            const double a = refOut[i], b = optOut[i];
            const double diff = b - a;
            const double scale = std::max({std::abs(a), std::abs(b), 1e-6});
            if (diff > worstAbove) worstAbove = diff;
            if (-diff > worstBelow) worstBelow = -diff;
            worstRel = std::max(worstRel, std::abs(diff) / scale);
            if (std::abs(diff) > 1e-12 * scale + 1e-15)
                ++mismatches;
            if (diff > 1e-9 * scale + 1e-12)
                ++unsafeLarger;
        }
        queries += qs.size();
    }

    if (queries == 0)
    {
        std::printf("no queries collected\n");
        return 1;
    }

    const double refUs = 1e6 * refSeconds / static_cast<double>(queries * reps);
    const double optUs = 1e6 * optSeconds / static_cast<double>(queries * reps);
    std::printf("\nhold-time certificate microbenchmark\n");
    std::printf("  problems %zu   queries %zu   reps %d\n", chosen.size(), queries, reps);
    std::printf("  reference   %8.3f us/query\n", refUs);
    std::printf("  optimized   %8.3f us/query   (%.2fx)\n", optUs, refUs / optUs);
    std::printf("  same search, original root solver: %8.3f us/query\n",
                1e6 * legacyRootSeconds / static_cast<double>(queries * reps));
    const double per = 1e6 / static_cast<double>(queries * reps);
    std::printf("  components: FK %.3f   worldRegion %.3f   certifiedRegion %.3f   "
                "HoldCache::build %.3f us\n",
                fkSeconds * per, geoSeconds * per, regionSeconds * per, buildSeconds * per);
    std::printf("              pair lever matvec %.3f   full 303-row screen %.3f us\n",
                matvecSeconds * per, screenSeconds * per);
    std::printf("  (sink %.3g)\n", sinkTotal);
    std::printf("  reference expensive rows/query: self %.2f  world %.2f\n",
                static_cast<double>(refSelfRows) / static_cast<double>(queries * reps),
                static_cast<double>(refWorldRows) / static_cast<double>(queries * reps));
    std::printf("  checksum ref %.12g  opt %.12g\n", refSum / reps, optSum / reps);
    std::printf("\nequivalence over %zu queries\n", queries);
    std::printf("  worst optimized - reference (unsafe direction) : %.3e\n", worstAbove);
    std::printf("  worst reference - optimized (conservative)     : %.3e\n", worstBelow);
    std::printf("  worst relative difference                      : %.3e\n", worstRel);
    std::printf("  differing answers %zu / %zu, of which larger %zu\n", mismatches, queries,
                unsafeLarger);
    std::printf("  screening-only arm (legacy root solver): worst |diff| %.3e over %zu "
                "differing of %zu\n", legacyWorst, legacyMismatches, queries);
    std::printf("  borrowed-geometry arm: worst |diff| %.3e over %zu differing of %zu\n",
                reuseWorst, reuseMismatches, queries);
    holdtime::reportCounters(stdout);
    return unsafeLarger == 0 ? 0 : 2;
}
