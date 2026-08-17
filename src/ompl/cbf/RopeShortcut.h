#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/Exception.h>

namespace ompl::cbf
{
    /// What one `ropeShortcut()` call did, for a caller that has to report on it.
    struct ShortcutReport
    {
        std::size_t anchors{0};    ///< nodes on the path after densification, at the end
        std::size_t screened{0};   ///< candidate pairs refused on geometry alone, at no QP cost
        std::size_t rollouts{0};   ///< candidate pairs that cost a CBF rollout
        std::size_t accepted{0};   ///< rollouts that became a shortcut
        std::size_t misses{0};     ///< input edges that were not on file and had to be re-rolled
        double lengthBefore{0.0};  ///< executed arc length in, radians
        double lengthAfter{0.0};   ///< executed arc length out, radians
        /// Largest gap, in radians, that an accepted shortcut had to be snapped across to
        /// finish exactly on its target. That final segment is the one part of the result
        /// no filter call certified, so this bounds the uncertified motion introduced.
        double maxArrivalGap{0.0};
    };

    namespace detail
    {
        using ShortcutConfiguration = FilteredStateSpace::Configuration;

        inline double arcLength(const std::vector<ShortcutConfiguration> &waypoints)
        {
            double total = 0.0;
            for (std::size_t k = 0; k + 1 < waypoints.size(); ++k)
                total += (waypoints[k + 1] - waypoints[k]).norm();
            return total;
        }

        template <typename Configuration>
        inline double vectorArcLength(const std::vector<Configuration> &waypoints)
        {
            double total = 0.0;
            for (std::size_t k = 0; k + 1 < waypoints.size(); ++k)
                total += (waypoints[k + 1] - waypoints[k]).norm();
            return total;
        }

        template <typename Configuration>
        inline void appendDistinct(std::vector<Configuration> &motion, const Configuration &q)
        {
            if (motion.empty() || (motion.back() - q).norm() > 1e-12)
                motion.push_back(q);
        }

        /// A dimension-independent representation used by the vector overload below.
        /// Each edge retains its complete filtered motion, so anchor densification never
        /// replaces a curved CBF rollout with a straight chord.
        template <typename Configuration>
        struct DenseMotion
        {
            std::vector<Configuration> anchors;
            std::vector<std::vector<Configuration>> edges;
        };

        template <typename Configuration>
        inline DenseMotion<Configuration> densifyMotion(const std::vector<Configuration> &motion,
                                                         double delta)
        {
            DenseMotion<Configuration> out;
            if (motion.empty())
                return out;
            out.anchors.push_back(motion.front());
            std::vector<Configuration> edge{motion.front()};
            double edgeLength = 0.0;

            for (std::size_t k = 1; k < motion.size(); ++k)
            {
                Configuration from = motion[k - 1];
                const Configuration to = motion[k];
                double remaining = (to - from).norm();
                while (remaining > 1e-12)
                {
                    const double needed = delta - edgeLength;
                    if (remaining + 1e-12 >= needed)
                    {
                        const Configuration cut = from + (needed / remaining) * (to - from);
                        appendDistinct(edge, cut);
                        out.edges.push_back(edge);
                        out.anchors.push_back(cut);
                        edge.assign(1, cut);
                        from = cut;
                        remaining = (to - from).norm();
                        edgeLength = 0.0;
                    }
                    else
                    {
                        appendDistinct(edge, to);
                        edgeLength += remaining;
                        remaining = 0.0;
                    }
                }
            }

            if ((out.anchors.back() - motion.back()).norm() > 1e-12)
            {
                appendDistinct(edge, motion.back());
                out.edges.push_back(edge);
                out.anchors.push_back(motion.back());
            }
            return out;
        }

        /// The executed motion of the edge \p from -> \p to: the recorded rollout if it is
        /// on file, otherwise a fresh one, which is what `executedPath()` also falls back
        /// to and is equally worth counting.
        inline std::vector<ShortcutConfiguration> edgeWaypoints(const FilteredStateSpace &space,
                                                                const ShortcutConfiguration &from,
                                                                const ShortcutConfiguration &to,
                                                                std::size_t *misses)
        {
            std::vector<ShortcutConfiguration> out;
            if (const FilteredStateSpace::EdgeRecord record = space.recordedEdge(from, to))
            {
                out.reserve(record.size());
                for (std::size_t k = 0; k < record.size(); ++k)
                    out.push_back(record[k]);
                return out;
            }

            if (misses != nullptr)
                ++*misses;
            FilteredStateSpace::Rollout rollout = space.roll(from, to, 1.0);
            out = std::move(rollout.waypoints);
            if (out.empty())
                out.push_back(from);
            out.back() = to;
            return out;
        }

        /// Cut one edge's executed motion into anchors roughly \p delta radians apart, and
        /// record each piece back into the ledger as an edge in its own right.
        ///
        /// This is what lets the shortcutter put nodes *inside* an edge without voiding the
        /// replay: an anchor sampled from a rollout is not, by itself, the endpoint of
        /// anything the ledger knows about, so `executedPath()` would miss it and re-roll.
        /// Recording the slices makes each one a real edge, and slicing is exact rather
        /// than approximate -- the rollout holds its control constant over a step, so the
        /// segment between two consecutive waypoints *is* the executed motion, and a cut
        /// strictly inside a step is a point on it.
        ///
        /// \p anchors must already end on `waypoints.front()`, and \p costs holds the
        /// cumulative arc length of \p anchors. Both are extended.
        inline void densifyEdge(const FilteredStateSpace &space,
                                const std::vector<ShortcutConfiguration> &waypoints, double delta,
                                std::vector<ShortcutConfiguration> &anchors, std::vector<double> &costs)
        {
            if (waypoints.size() < 2)
                return;

            std::vector<ShortcutConfiguration> slice{waypoints.front()};
            double length = 0.0;

            const auto cut = [&](const ShortcutConfiguration &at)
            {
                // `record()` refuses both of these anyway; refusing them here too keeps the
                // anchor list free of a node the ledger has no edge for.
                if (slice.size() < 2 || FilteredStateSpace::bitwiseEqual(slice.front(), at))
                    return;
                space.record(slice.front(), at, slice);
                anchors.push_back(at);
                costs.push_back(costs.back() + length);
                slice.assign(1, at);
                length = 0.0;
            };

            for (std::size_t k = 1; k < waypoints.size(); ++k)
            {
                slice.push_back(waypoints[k]);
                length += (waypoints[k] - waypoints[k - 1]).norm();
                // The edge's own endpoint is always a cut: it is a state the path holds.
                if (length >= delta || k + 1 == waypoints.size())
                    cut(waypoints[k]);
            }
        }
    }  // namespace detail

    /// Shorten \p path in place, RRT-Rope style, with the CBF rollout as the edge test.
    ///
    /// Same shape as `geometric::PathSimplifier::ropeShortcutPath()` -- densify to \p delta,
    /// then try to connect each node to the farthest node it can reach, restarting the
    /// sweep whenever that succeeds -- with the three things rope assumes about a straight
    /// line replaced:
    ///
    /// - Densification walks the recorded rollout rather than calling the state space's
    ///   `interpolate()`, which on a `FilteredStateSpace` is itself a rollout.
    /// - An edge is tested by rolling it and requiring arrival, not by checking a segment.
    /// - A segment costs its executed arc length, not the Euclidean distance across it: a
    ///   deflected edge is longer to drive than the straight line it was aimed along, and
    ///   costing it Euclidean would buy shortcuts that are longer than what they replace.
    ///
    /// \p delta is the anchor spacing in radians; it sets both how finely the path can be
    /// cut and, through the number of anchors, the quadratic cost of the sweep.
    /// \p equivalenceTolerance is relative to \p delta, as in rope.
    /// \p arrivalTolerance is how close a rollout must finish to its target before the
    /// remainder is snapped away, in radians; non-positive derives a twentieth of the
    /// space's `reachTolerance()`. This is deliberately far tighter than the space's own
    /// tolerance -- see the arrival loop below for why.
    ///
    /// Every edge of the result is in the ledger, so `executedPath()` still replays the
    /// whole path with `misses == 0`.
    ///
    /// Returns true if the path changed.
    inline bool ropeShortcut(geometric::PathGeometric &path, double delta,
                             double equivalenceTolerance = 0.1, ShortcutReport *report = nullptr,
                             double arrivalTolerance = 0.0)
    {
        using Configuration = FilteredStateSpace::Configuration;

        const base::SpaceInformationPtr &si = path.getSpaceInformation();
        const auto *space = dynamic_cast<const FilteredStateSpace *>(si->getStateSpace().get());
        if (space == nullptr)
            throw Exception("ropeShortcut requires a FilteredStateSpace");

        if (report != nullptr)
            *report = ShortcutReport();
        if (delta <= 0.0 || path.getStateCount() < 3)
            return false;

        std::size_t misses = 0;
        // A twentieth of a step of travel: ~3 mrad here, under 3 mm at the arm's longest
        // lever, which is small against any margin the barrier is guarding.
        const double arrival =
            arrivalTolerance > 0.0 ? arrivalTolerance : 0.05 * space->reachTolerance();
        // Enough attempts to converge when the target is reachable, few enough that a
        // stalled candidate is cheap to give up on.
        constexpr unsigned int arrivalAttempts = 4;

        // Densify. Every anchor is an endpoint of a recorded edge from here on, and
        // costs[i] is the executed arc length from the start of the path to anchor i.
        std::vector<Configuration> anchors{FilteredStateSpace::configurationOf(path.getState(0))};
        std::vector<double> costs{0.0};
        for (std::size_t i = 0; i + 1 < path.getStateCount(); ++i)
        {
            const Configuration from = FilteredStateSpace::configurationOf(path.getState(i));
            const Configuration to = FilteredStateSpace::configurationOf(path.getState(i + 1));
            detail::densifyEdge(*space, detail::edgeWaypoints(*space, from, to, &misses), delta, anchors,
                                costs);
        }

        const double lengthBefore = costs.back();
        std::size_t screened = 0;
        std::size_t rollouts = 0;
        std::size_t accepted = 0;
        double maxGap = 0.0;
        bool changed = false;

        // The sweep. Farthest j first, restart from the beginning after every shortcut,
        // stop as soon as one reaches the end of the path -- all as in rope.
        std::size_t i = 0;
        while (i + 2 < anchors.size())
        {
            bool restart = false;
            bool finished = false;

            for (std::size_t j = anchors.size() - 1; j > i + 1; --j)
            {
                const double alongPath = costs[j] - costs[i];
                const double euclid = (anchors[j] - anchors[i]).norm();

                // The straight line is a lower bound on what any rollout between these two
                // can travel, so this refuses a candidate that cannot win without paying
                // for a single QP solve.
                if (euclid >= alongPath)
                {
                    ++screened;
                    continue;
                }

                // ...and by the same bound, `alongPath - euclid` is the most a shortcut
                // could ever save here. Below the tolerance this stretch is already as
                // straight as it can be made. Rope tests this only after checking the
                // edge; testing it first is sound and skips the rollout entirely.
                if (alongPath - euclid < equivalenceTolerance * delta)
                {
                    ++screened;
                    if (j + 1 == anchors.size())
                        finished = true;
                    break;
                }

                if (FilteredStateSpace::bitwiseEqual(anchors[i], anchors[j]))
                {
                    ++screened;
                    continue;
                }

                FilteredStateSpace::Rollout rollout = space->roll(anchors[i], anchors[j], 1.0);
                ++rollouts;
                if (!rollout.reachedTarget || rollout.waypoints.size() < 2)
                    continue;

                // `reachedTarget` means "within `reachTolerance`", and the planner closes
                // that remainder by snapping its last waypoint onto the target. Here that
                // would be unsound at a rate the planner never sees: the shortcutter takes
                // this path on *every* edge it installs, and `reachTolerance` is a full
                // step of travel -- measured at up to 0.059 rad on the shelf scene, which
                // is a straight jump no filter call certified, through the region the
                // rollout was deflecting around in the first place.
                //
                // So close the distance with more rollout instead of teleporting across
                // it, and take the residual snap only once it is negligible. Re-aiming
                // from where the last attempt stalled is what makes progress: each roll
                // gets a fresh horizon for the distance that is actually left.
                std::vector<Configuration> motion = std::move(rollout.waypoints);
                double gap = (motion.back() - anchors[j]).norm();
                for (unsigned int attempt = 0; attempt < arrivalAttempts && gap > arrival;
                     ++attempt)
                {
                    const FilteredStateSpace::Rollout closing =
                        space->roll(motion.back(), anchors[j], 1.0);
                    ++rollouts;
                    const double closed = (closing.end - anchors[j]).norm();
                    // Stalled: another attempt would re-roll the same trajectory from the
                    // same state and stall the same way.
                    if (closing.waypoints.size() < 2 || closed >= gap)
                        break;
                    motion.insert(motion.end(), closing.waypoints.begin() + 1,
                                  closing.waypoints.end());
                    gap = closed;
                }
                // Never arrived. Rejecting costs nothing -- the path simply stays as it
                // was -- which is exactly why the shortcutter can afford to hold a
                // standard the planner cannot.
                if (gap > arrival)
                    continue;

                maxGap = std::max(maxGap, gap);
                motion.back() = anchors[j];
                // Held to the same tolerance the screen above uses, rather than to merely
                // being shorter. That is what bounds the sweep: every accepted shortcut
                // takes at least `equivalenceTolerance * delta` radians off the path, so
                // the restart-from-the-top loop can run at most `length / (tolerance *
                // delta)` times before there is nothing left to win.
                if (detail::arcLength(motion) > alongPath - equivalenceTolerance * delta)
                    continue;

                // Splice: head up to i, the new edge densified in place of (i, j), then the
                // untouched tail. Recording the slices is all the recording needed -- the
                // path never holds the edge i -> j whole, so storing it too would only
                // consume ledger capacity for a lookup nobody makes.
                std::vector<Configuration> spliced(anchors.begin(), anchors.begin() + i + 1);
                std::vector<double> splicedCosts(costs.begin(), costs.begin() + i + 1);
                detail::densifyEdge(*space, motion, delta, spliced, splicedCosts);
                for (std::size_t k = j + 1; k < anchors.size(); ++k)
                {
                    spliced.push_back(anchors[k]);
                    splicedCosts.push_back(splicedCosts.back() + (costs[k] - costs[k - 1]));
                }

                const bool reachedEnd = (j + 1 == anchors.size());
                anchors = std::move(spliced);
                costs = std::move(splicedCosts);
                ++accepted;
                changed = true;

                if (reachedEnd)
                    finished = true;
                else
                    restart = true;
                break;
            }

            if (finished)
                break;
            i = restart ? 0 : i + 1;
        }

        if (report != nullptr)
        {
            report->anchors = anchors.size();
            report->screened = screened;
            report->rollouts = rollouts;
            report->accepted = accepted;
            report->misses = misses;
            report->lengthBefore = lengthBefore;
            report->lengthAfter = costs.back();
            report->maxArrivalGap = maxGap;
        }

        // Write the anchors back, reusing the states the path already owns.
        std::vector<base::State *> &states = path.getStates();
        while (states.size() > anchors.size())
        {
            si->freeState(states.back());
            states.pop_back();
        }
        while (states.size() < anchors.size())
            states.push_back(si->allocState());
        for (std::size_t k = 0; k < anchors.size(); ++k)
            FilteredStateSpace::setState(states[k], anchors[k]);

        return changed;
    }

    /// Dimension-independent CBF RRT-Rope over an already executed motion.
    ///
    /// This overload carries the same algorithm and safety contract as the
    /// `PathGeometric` overload, but accepts plain Eigen-like configurations and a
    /// caller-provided rollout function. It is intended for control-space planners
    /// and robots whose configuration dimension is not `FilteredStateSpace`'s legacy
    /// UR5 dimension. `rollout(from, to)` must return every filtered integration
    /// waypoint, beginning at `from`; an empty vector rejects the candidate.
    template <typename Configuration, typename RolloutFunction>
    std::vector<Configuration> ropeShortcut(const std::vector<Configuration> &input, double delta,
                                            double arrivalTolerance, RolloutFunction &&rollout,
                                            ShortcutReport *report = nullptr,
                                            double equivalenceTolerance = 0.1)
    {
        if (report != nullptr)
            *report = ShortcutReport();
        const double lengthBefore = detail::vectorArcLength(input);
        if (input.size() < 3 || delta <= 0.0)
        {
            if (report != nullptr)
                report->lengthBefore = report->lengthAfter = lengthBefore;
            return input;
        }
        if (arrivalTolerance <= 0.0)
            throw Exception("ropeShortcut: arrival tolerance must be positive");

        detail::DenseMotion<Configuration> path = detail::densifyMotion(input, delta);
        constexpr unsigned int arrivalAttempts = 4;
        std::size_t screened = 0;
        std::size_t rollouts = 0;
        std::size_t accepted = 0;
        double maxGap = 0.0;

        std::size_t i = 0;
        while (i + 2 < path.anchors.size())
        {
            bool restart = false;
            bool finished = false;
            for (std::size_t j = path.anchors.size() - 1; j > i + 1; --j)
            {
                double along = 0.0;
                for (std::size_t edge = i; edge < j; ++edge)
                    along += detail::vectorArcLength(path.edges[edge]);
                const double euclidean = (path.anchors[j] - path.anchors[i]).norm();
                if (euclidean >= along)
                {
                    ++screened;
                    continue;
                }
                if (along - euclidean < equivalenceTolerance * delta)
                {
                    ++screened;
                    if (j + 1 == path.anchors.size())
                        finished = true;
                    break;
                }

                std::vector<Configuration> motion = rollout(path.anchors[i], path.anchors[j]);
                ++rollouts;
                if (motion.size() < 2)
                    continue;

                double gap = (motion.back() - path.anchors[j]).norm();
                for (unsigned int attempt = 0; attempt < arrivalAttempts && gap > arrivalTolerance;
                     ++attempt)
                {
                    std::vector<Configuration> closing = rollout(motion.back(), path.anchors[j]);
                    ++rollouts;
                    if (closing.size() < 2)
                        break;
                    const double closed = (closing.back() - path.anchors[j]).norm();
                    if (closed >= gap)
                        break;
                    motion.insert(motion.end(), closing.begin() + 1, closing.end());
                    gap = closed;
                }
                if (gap > arrivalTolerance)
                    continue;

                maxGap = std::max(maxGap, gap);
                motion.back() = path.anchors[j];
                if (detail::vectorArcLength(motion) > along - equivalenceTolerance * delta)
                    continue;

                const bool reachedEnd = (j + 1 == path.anchors.size());
                detail::DenseMotion<Configuration> replacement = detail::densifyMotion(motion, delta);
                detail::DenseMotion<Configuration> spliced;
                spliced.anchors.insert(spliced.anchors.end(), path.anchors.begin(),
                                       path.anchors.begin() + i + 1);
                spliced.edges.insert(spliced.edges.end(), path.edges.begin(), path.edges.begin() + i);
                spliced.anchors.insert(spliced.anchors.end(), replacement.anchors.begin() + 1,
                                       replacement.anchors.end());
                spliced.edges.insert(spliced.edges.end(), replacement.edges.begin(),
                                     replacement.edges.end());
                spliced.anchors.insert(spliced.anchors.end(), path.anchors.begin() + j + 1,
                                       path.anchors.end());
                spliced.edges.insert(spliced.edges.end(), path.edges.begin() + j, path.edges.end());
                path = std::move(spliced);
                ++accepted;

                if (reachedEnd)
                    finished = true;
                else
                    restart = true;
                break;
            }
            if (finished)
                break;
            i = restart ? 0 : i + 1;
        }

        std::vector<Configuration> output;
        if (!path.anchors.empty())
            output.push_back(path.anchors.front());
        for (const auto &edge : path.edges)
            for (std::size_t k = 1; k < edge.size(); ++k)
                detail::appendDistinct(output, edge[k]);

        if (report != nullptr)
        {
            report->anchors = path.anchors.size();
            report->screened = screened;
            report->rollouts = rollouts;
            report->accepted = accepted;
            report->lengthBefore = lengthBefore;
            report->lengthAfter = detail::vectorArcLength(output);
            report->maxArrivalGap = maxGap;
        }
        return output;
    }
}  // namespace ompl::cbf
