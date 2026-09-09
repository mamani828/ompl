#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/util/Exception.h>

namespace ompl::cbf
{
    /// A path-length objective that charges an edge what it actually cost to drive,
    /// not what the straight line between its endpoints would have cost.
    ///
    /// `RobotFilteredStateSpace::distance()` is Euclidean, and is only a *lower bound*
    /// on the length of a CBF rollout that deflected around an obstacle -- see that
    /// class's header. `PathLengthOptimizationObjective`, and everything else in OMPL
    /// that reports path cost, uses `distance()` as the edge cost by default, so an
    /// optimizing planner over a filtered space would otherwise optimize the wrong
    /// number: it reports a solution as cheaper than the motion the robot will
    /// actually run, and prunes/orders its search on that same wrong number.
    /// `motionCostHeuristic()` is left at the base class's `distance()`: that is still a
    /// valid lower bound -- a straight line is never longer than a deflected path to the
    /// same target -- so anything relying on it for admissible pruning keeps working,
    /// only more conservatively near contact.
    ///
    /// ### `motionCost()` has to do the rollout itself, not merely read it back
    ///
    /// The obvious version of this override reads the executed length off the ledger and
    /// falls back to `distance()` when the edge is not on file yet, on the theory that
    /// the fallback only fires before validation and self-corrects once
    /// `FilteredMotionValidator::checkMotion()` has recorded the edge. That theory is
    /// wrong for at least this BIT* (`BITstar.cpp`): `SearchQueue`'s edge processing
    /// computes `CostHelper::trueEdgeCost()` -- this objective's `motionCost()` -- and
    /// hands the result to `addEdge()` *before* `checkEdge()`/`checkMotion()` ever runs
    /// (`BITstar.cpp:562` precedes `:572`), so "not recorded yet" is not a rare corner
    /// case here, it is the state of affairs the *first* time every single edge is ever
    /// costed. Worse, a vertex that keeps the parent it was first given is a "freebie"
    /// on later visits (`BITstar.cpp:540`) and is never re-costed at all, so a first
    /// pricing taken as the Euclidean fallback can stay wrong for the life of the run,
    /// not just until the next query.
    ///
    /// So this override does not wait to be asked again: when there is no ledger entry,
    /// it runs the rollout itself, right here, and only when that rollout actually
    /// arrives -- the same bar `FilteredMotionValidator`'s fallback holds it to --
    /// commits it to the ledger. `checkMotion()`'s own rollout moments later then finds
    /// the edge already on file and is served for free instead of re-rolling, so the
    /// eager query is not pure duplicated work; the duplication that *is* real is any
    /// edge BIT* prices but never carries to `checkEdge()` because a cheaper alternative
    /// wins first (`SearchQueue`'s heuristic gates at `BITstar.cpp:550,556` sit in front
    /// of `trueEdgeCost()`, so this is already the minority of what BIT* looks at, not
    /// all of it, but it is not zero). A rollout that does not arrive is not recorded --
    /// recording it would tell `checkMotion()` a non-arriving edge is valid -- and this
    /// objective falls back to the same Euclidean estimate for it, which is safe because
    /// `checkEdge()` is the one that will actually reject it a moment later.
    ///
    /// Set \p eager to false to get the naive, provably-wrong version instead; it exists
    /// to measure the bug this class works around, not as a mode anyone should plan
    /// with.
    template <typename Robot>
    class RobotFilteredPathLengthObjective : public base::PathLengthOptimizationObjective
    {
    public:
        using Space = RobotFilteredStateSpace<Robot>;

        explicit RobotFilteredPathLengthObjective(const base::SpaceInformationPtr &si, bool eager = true)
          : base::PathLengthOptimizationObjective(si), space_(spaceOf(si.get())), eager_(eager)
        {
        }

        base::Cost motionCost(const base::State *s1, const base::State *s2) const override
        {
            if (const typename Space::EdgeRecord record = space_->recordedEdge(s1, s2))
                return base::Cost(lengthOf(record));

            if (!eager_)
                return base::PathLengthOptimizationObjective::motionCost(s1, s2);

            typename Space::Rollout rollout = space_->roll(s1, s2, 1.0);
            if (!rollout.reachedTarget)
                return base::PathLengthOptimizationObjective::motionCost(s1, s2);

            const double length = lengthOf(rollout.waypoints);
            space_->record(Space::configurationOf(s1), Space::configurationOf(s2),
                           std::move(rollout.waypoints));
            return base::Cost(length);
        }

    private:
        double lengthOf(const typename Space::EdgeRecord &record) const
        {
            double length = 0.0;
            for (std::size_t i = 0; i + 1 < record.size(); ++i)
                length += Space::Operations::distance(record[i], record[i + 1], space_->maxSpeed());
            return length;
        }

        double lengthOf(const std::vector<typename Space::Configuration> &waypoints) const
        {
            double length = 0.0;
            for (std::size_t i = 0; i + 1 < waypoints.size(); ++i)
                length +=
                    Space::Operations::distance(waypoints[i], waypoints[i + 1], space_->maxSpeed());
            return length;
        }

        static const Space *spaceOf(const base::SpaceInformation *si)
        {
            const auto *space = dynamic_cast<const Space *>(si->getStateSpace().get());
            if (space == nullptr)
                throw Exception(
                    "RobotFilteredPathLengthObjective requires a matching RobotFilteredStateSpace");
            return space;
        }

        const Space *space_;
        bool eager_;
    };

    /// Compatibility alias for the original UR5 API.
    using FilteredPathLengthObjective = RobotFilteredPathLengthObjective<robots::UR5>;
}  // namespace ompl::cbf
