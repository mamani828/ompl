#pragma once
#include "HoldTimeCertificate.h"
#include <ompl/cbf/FilteredStateSpace.h>
#include <algorithm>
#include <limits>
using UR5 = ompl::robots::UR5;
using Barrier = ompl::cbf::ClearanceBarrier;
using Space = ompl::cbf::FilteredStateSpace;
using Configuration = UR5::Configuration;
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
