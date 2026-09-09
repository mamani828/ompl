// demos/SSTFiltered.h
//
// `geometric::SST` with the three defects that stop it being usable over
// `cbf::FilteredStateSpace` fixed, and nothing else changed. Kept out of the shared
// library on the same grounds as RRTConnectDD.h: this is a comparison row, and the fix
// in (1) below changes stock SST's behaviour on every problem, not just filtered ones.
//
// Why SST at all. The BIT* spike (UR5BITStarSpike.cpp) found that an optimizing planner
// whose vertices are a pre-sampled batch fights the rollout: every candidate edge asks
// the rollout to *arrive exactly* at an already-chosen vertex, so a perfectly good
// deflected rollout is simply a rejected edge, paid for with a full rollout instead of a
// handful of discrete checks. SST grows the opposite way. `monteCarloProp` samples a
// direction and a length, calls `interpolate()`, and keeps *wherever it landed* as the
// node -- which is exactly what `FilteredStateSpace` promises ("the deflected endpoint is
// a perfectly good new tree node"). So SST is the optimizing planner that does not need
// the rollout to arrive anywhere in particular.
//
// The three fixes:
//
// 1. **Pruning never happens in stock SST.** `Motion::inactive_` is initialised false and
//    is only ever assigned inside `while (oldRep->inactive_ && ...)` (`SST.cpp:362-364`,
//    and identically in `control/SST.cpp:353-355`) -- a loop whose body sets the
//    condition it just tested, so the body is unreachable and no node is ever marked
//    inactive, removed from the tree, or skipped by `selectNode`'s `!inactive_` filter.
//    What survives is the *admission* gate: at most one node per witness cell is ever
//    added. What is lost is the removal of superseded representatives, which linger in
//    the nearest-neighbour structure and stay selectable forever. Since a bounded, sparse
//    tree is the whole reason to reach for SST here, `prune()` below marks the dominated
//    representative inactive *before* the cascade, which is plainly what the loop was
//    written to do.
//
//    That the cascade only ever deletes *inactive* nodes is what makes it safe, and the
//    reason is worth writing down because it is not obvious: witnesses hold a raw
//    `rep_` pointer, and the cascade walks up the parent chain, so deleting a node some
//    other witness still represents would be a use-after-free. It cannot happen. A node
//    is marked inactive exactly when its witness has just relinked away from it, so
//    "inactive" means "no witness refers to this", and the cascade's own condition
//    refuses to step into a node that is not inactive. Nodes on a live parent chain are
//    likewise safe: they have a child, so the `numChildren_ == 0` test stops there.
//
// 2. **`monteCarloProp` leaks a state per rejected extension.** It allocates
//    (`SST.cpp:205`) and stock frees only inside the accepted branch (`SST.cpp:303`,
//    nested under the witness test at `:294`), so every extension that fails
//    `checkMotion` or loses to a representative leaks. Pre-existing, but it bites harder
//    over a filtered space, which rejects more often: an extension that makes no headway
//    is reported as going nowhere (`FilteredStateSpace::interpolate`'s progress test) and
//    is then refused as zero-length. `propagateInto()` writes into a caller-owned state
//    and allocates nothing.
//
// 3. **`selectNode`'s fallback cannot terminate once nodes can be inactive.** It widens
//    `k` by 5 until it finds an active node, with no bound (`SST.cpp:165-173`). That is
//    unreachable in stock SST -- nothing is ever inactive -- and reachable here.
//    `selectActive()` bounds the widening by the tree size and then takes the nearest
//    node whatever its state, an inactive node being a worse parent rather than an
//    unusable one: it is still in the tree, so its state is still alive and still safe.
//
// What is deliberately *not* changed, so that this row measures SST rather than a variant
// of it -- these are the next row's subject:
//
// - When the sampled target is already within `range`, stock takes it as the target
//   directly (`SST.cpp:274-280` only interpolates when `d > maxDistance_`), so the edge
//   asks the rollout to arrive at a raw sample and gets the BIT*-shaped outcome. Counted
//   as `rawTargets` below, which is the number the next row has to beat.
// - The step length stays `U(0, range]` rather than coming from the filter's certified
//   duration, and the pruning radius stays uniform rather than scaling with clearance.
// - The 50/50 steer-or-propagate coin toss throws away the goal-biased sample half the
//   time it draws one (`SST.cpp:272` overwrites `attemptToReachGoal`).
//
// ### Wiring
//
// Costs must come from `cbf::RobotFilteredPathLengthObjective`, or selection and witness
// domination are decided on Euclidean `distance()`, which is a lower bound on what a
// deflected rollout actually costs -- and here that mispricing does not merely misreport
// a solution, it *deletes* nodes. `eager = false` is enough, unlike BIT*: SST prices an
// edge at `SST.cpp:290`, immediately after `checkMotion()` succeeded at `:288`, so the
// edge is already on file and the eager rollout would never fire.
//
// All three radii must be set. The defaults (`range 5`, `selectionRadius 5`,
// `pruningRadius 3`) are in radians of Euclidean joint distance and are nonsense for an
// arm: a 5 rad selection radius covers most of the tree, and a 3 rad pruning radius
// admits about one node.

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/sst/SST.h>

namespace ompl::geometric
{
    class SSTFiltered : public SST
    {
    public:
        /// Where an iteration went, so a row can be costed without instrumenting the
        /// filter. `rejected + dominated + admitted == iterations`, less the iterations
        /// that found no selectable node at all.
        struct Statistics
        {
            std::size_t iterations{0};
            std::size_t steered{0};     ///< extensions aimed at the sampled target
            std::size_t propagated{0};  ///< extensions aimed at a fresh uniform direction
            /// Steered extensions whose target was the raw sample rather than a rolled
            /// endpoint, so the edge had to *arrive* rather than land where it landed.
            /// See "What is deliberately not changed".
            std::size_t rawTargets{0};
            std::size_t rejected{0};   ///< `checkMotion` said no
            std::size_t dominated{0};  ///< valid, but the witness kept its representative
            std::size_t admitted{0};   ///< became a tree node
            std::size_t deactivated{0};  ///< representatives superseded by a cheaper node
            std::size_t pruned{0};       ///< of those, how many were deleted outright
        };

        explicit SSTFiltered(const base::SpaceInformationPtr &si) : SST(si)
        {
            setName("SSTFiltered");
            Planner::declareParam<bool>("pruning", this, &SSTFiltered::setPruning,
                                        &SSTFiltered::getPruning, "0,1");
        }

        /// Whether a superseded representative is marked inactive and removed. On is
        /// SST as published; off reproduces stock OMPL's behaviour, where the cascade is
        /// dead code. This is the A/B for fix (1).
        void setPruning(bool pruning)
        {
            pruning_ = pruning;
        }

        bool getPruning() const
        {
            return pruning_;
        }

        const Statistics &statistics() const
        {
            return statistics_;
        }

        void clear() override
        {
            SST::clear();
            statistics_ = Statistics();
        }

        base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override
        {
            checkValidity();
            base::Goal *goal = pdef_->getGoal().get();
            auto *goalSampler = dynamic_cast<base::GoalSampleableRegion *>(goal);

            while (const base::State *st = pis_.nextStart())
            {
                auto *motion = new Motion(si_);
                si_->copyState(motion->state_, st);
                motion->accCost_ = opt_->identityCost();
                nn_->add(motion);
                findClosestWitness(motion);
            }

            if (nn_->size() == 0)
            {
                OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
                return base::PlannerStatus::INVALID_START;
            }

            if (!sampler_)
                sampler_ = si_->allocStateSampler();

            const base::ReportIntermediateSolutionFn intermediateSolutionCallback =
                pdef_->getIntermediateSolutionCallback();

            OMPL_INFORM("%s: Starting planning with %u states already in datastructure",
                        getName().c_str(), nn_->size());

            Motion *solution = nullptr;
            Motion *approxsol = nullptr;
            double approxdif = std::numeric_limits<double>::infinity();
            bool sufficientlyShort = false;

            // The scratch node the nearest-neighbour and witness queries are phrased
            // against, and the buffer every extension is grown into. Two allocations for
            // the whole run; stock allocates one state per propagated extension and frees
            // some of them.
            auto *rmotion = new Motion(si_);
            base::State *rstate = rmotion->state_;
            base::State *xstate = si_->allocState();

            while (ptc == false)
            {
                ++statistics_.iterations;

                if (goalSampler != nullptr && rng_.uniform01() < goalBias_ && goalSampler->canSample())
                    goalSampler->sampleGoal(rstate);
                else
                    sampler_->sampleUniform(rstate);

                Motion *nmotion = selectActive(rmotion);
                if (nmotion == nullptr)
                    break;  // an empty tree, which the start loop above rules out

                if (rng_.uniform01() < 0.5)
                {
                    ++statistics_.steered;
                    const double d = si_->distance(nmotion->state_, rstate);
                    if (d > maxDistance_)
                    {
                        // The rollout decides where this lands; `xstate` is wherever it
                        // got to, which is the state that becomes the node.
                        si_->getStateSpace()->interpolate(nmotion->state_, rstate,
                                                          maxDistance_ / d, xstate);
                        si_->copyState(rstate, xstate);
                    }
                    else
                    {
                        // Target is the sample itself: no rollout has been run, so the
                        // edge is held to arrive-or-reject.
                        ++statistics_.rawTargets;
                    }
                }
                else
                {
                    ++statistics_.propagated;
                    propagateInto(nmotion, xstate);
                    si_->copyState(rstate, xstate);
                }

                // No state validity checker is consulted anywhere: over a filtered space
                // the barrier certified every state of the rollout as it was produced,
                // and this asks only whether the edge is one the space actually produced.
                if (!si_->checkMotion(nmotion->state_, rstate))
                {
                    ++statistics_.rejected;
                    continue;
                }

                const base::Cost incCost = opt_->motionCost(nmotion->state_, rstate);
                const base::Cost cost = opt_->combineCosts(nmotion->accCost_, incCost);
                Witness *closestWitness = findClosestWitness(rmotion);

                // `rep_ == rmotion` means the witness was just created for this state and
                // linked to the scratch node, so there is no incumbent to beat.
                if (closestWitness->rep_ != rmotion &&
                    !opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_))
                {
                    ++statistics_.dominated;
                    continue;
                }

                Motion *oldRep = closestWitness->rep_;
                auto *motion = new Motion(si_);
                motion->accCost_ = cost;
                si_->copyState(motion->state_, rstate);
                motion->parent_ = nmotion;
                nmotion->numChildren_++;
                closestWitness->linkRep(motion);
                nn_->add(motion);
                ++statistics_.admitted;

                double dist = 0.0;
                const bool solved = goal->isSatisfied(motion->state_, &dist);
                if (solved && opt_->isCostBetterThan(motion->accCost_, prevSolutionCost_))
                {
                    approxdif = dist;
                    solution = motion;
                    recordSolution(solution);
                    prevSolutionCost_ = solution->accCost_;

                    OMPL_INFORM("Found solution with cost %.2f", solution->accCost_.value());
                    if (intermediateSolutionCallback)
                    {
                        const std::vector<const base::State *> states(prevSolution_.begin(),
                                                                      prevSolution_.end());
                        intermediateSolutionCallback(this, states, prevSolutionCost_);
                    }
                    sufficientlyShort = opt_->isSatisfied(solution->accCost_);
                    if (sufficientlyShort)
                        break;
                }
                if (solution == nullptr && dist < approxdif)
                {
                    approxdif = dist;
                    approxsol = motion;
                    recordSolution(approxsol);
                }

                if (oldRep != rmotion)
                    prune(oldRep);
            }

            bool solved = false;
            bool approximate = false;
            if (solution == nullptr)
            {
                solution = approxsol;
                approximate = true;
            }

            // Note both pointers may by now name a deleted node: a solution leaf has no
            // children, so a later cheaper node in its witness cell can prune it. Neither
            // is dereferenced here, and the path is built from the cloned states taken
            // when the solution was found, so the path outlives its nodes. The edges do
            // too -- the ledger is keyed by configuration, not by tree node.
            if (solution != nullptr)
            {
                auto path(std::make_shared<PathGeometric>(si_));
                for (int i = static_cast<int>(prevSolution_.size()) - 1; i >= 0; --i)
                    path->append(prevSolution_[i]);
                solved = true;
                pdef_->addSolutionPath(path, approximate, approxdif, getName());
            }

            si_->freeState(xstate);
            if (rmotion->state_ != nullptr)
                si_->freeState(rmotion->state_);
            rmotion->state_ = nullptr;
            delete rmotion;

            OMPL_INFORM("%s: Created %u states in %zu iterations", getName().c_str(), nn_->size(),
                        statistics_.iterations);

            return {solved, approximate};
        }

    private:
        /// Grow toward a fresh uniform direction, into a caller-owned state. Stock's
        /// `monteCarloProp` with the allocation removed -- see fix (2).
        void propagateInto(const Motion *from, base::State *out)
        {
            sampler_->sampleUniform(out);
            const double step = rng_.uniformReal(0, maxDistance_);
            const double d = si_->distance(from->state_, out);
            if (d > std::numeric_limits<double>::epsilon())
                si_->getStateSpace()->interpolate(from->state_, out, step / d, out);
            si_->enforceBounds(out);
        }

        /// The cheapest selectable node near \p sample. See fix (3) for how this differs
        /// from `SST::selectNode`.
        Motion *selectActive(Motion *sample)
        {
            std::vector<Motion *> near;
            Motion *selected = nullptr;
            base::Cost best = opt_->infiniteCost();

            nn_->nearestR(sample, selectionRadius_, near);
            for (Motion *candidate : near)
            {
                if (!candidate->inactive_ && opt_->isCostBetterThan(candidate->accCost_, best))
                {
                    best = candidate->accCost_;
                    selected = candidate;
                }
            }
            if (selected != nullptr)
                return selected;

            const std::size_t size = nn_->size();
            for (std::size_t k = 1; selected == nullptr && k <= size; k += 5)
            {
                nn_->nearestK(sample, k, near);
                for (Motion *candidate : near)
                {
                    if (!candidate->inactive_)
                    {
                        selected = candidate;
                        break;
                    }
                }
            }
            if (selected == nullptr && size > 0)
                selected = nn_->nearest(sample);
            return selected;
        }

        /// Retire a representative that has just been superseded, and delete it and any
        /// ancestor it was the last child of. See fix (1) for why only inactive nodes may
        /// be deleted.
        void prune(Motion *oldRep)
        {
            if (!pruning_)
                return;

            oldRep->inactive_ = true;
            ++statistics_.deactivated;

            // `parent_ != nullptr` keeps the cascade off a start node, whose parent
            // pointer would be dereferenced for its child count. A start node cannot be
            // dominated under a non-negative cost -- nothing beats `identityCost` -- so
            // this guards an unreachable case rather than a real one.
            while (oldRep->inactive_ && oldRep->numChildren_ == 0 && oldRep->parent_ != nullptr)
            {
                nn_->remove(oldRep);
                if (oldRep->state_ != nullptr)
                    si_->freeState(oldRep->state_);
                oldRep->state_ = nullptr;

                Motion *parent = oldRep->parent_;
                parent->numChildren_--;
                delete oldRep;
                ++statistics_.pruned;
                oldRep = parent;
            }
        }

        /// Snapshot the branch ending at \p leaf as cloned states. Every node on a live
        /// parent chain has a child, so none of them can have been pruned.
        void recordSolution(Motion *leaf)
        {
            for (base::State *state : prevSolution_)
                if (state != nullptr)
                    si_->freeState(state);
            prevSolution_.clear();
            for (Motion *node = leaf; node != nullptr; node = node->parent_)
                prevSolution_.push_back(si_->cloneState(node->state_));
        }

        bool pruning_{true};
        Statistics statistics_;
    };
}  // namespace ompl::geometric
