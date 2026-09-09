// demos/RRTConnectDD.h
//
// RRTConnect plus the two tree-growth heuristics VAMP's own rrtc.hh has on by default
// and stock ompl::geometric::RRTConnect does not: dynamic-domain sampling (Yershova et
// al.) and ratio-gated tree balancing. Kept out of the shared library deliberately --
// this is a comparison tool for the VAMP-vs-OMPL sample-efficiency question, not a
// general-purpose planner, and touching geometric::RRTConnect itself would change
// behaviour for every other demo and test that uses it.
//
// Both additions are ported to match vamp::planning::RRTCSettings' formulas as closely
// as OMPL's iteration structure allows, not approximated:
//
// - Dynamic domain: each tree node carries a radius (unset = infinite). A random sample
//   farther from its nearest node than that node's radius is rejected before any
//   extension is attempted -- and the rejection touches no radius, exactly as in
//   rrtc.hh, since nothing was learned from a sample that was never tried. A node's
//   radius is only ever touched once it has actually been extended toward: it grows by
//   `(1 + alpha)` on a successful extension (but only if already finite -- a node that
//   has never failed stays unconstrained), and on a colliding extension it is
//   initialised to `radius` if unset or shrunk by `(1 - alpha)`, floored at
//   `minRadius`, if not. This only gates the *first* extension of an iteration (growing
//   toward a fresh random sample); the connect-phase extensions that follow are
//   deliberately ungated, matching rrtc.hh's `n_extensions` loop.
//
// - Tree balance: rather than swapping which tree grows every iteration
//   unconditionally, the swap is gated on `ratio = |priorGrowerSize - priorTargetSize|
//   / priorGrowerSize < treeRatio` (or unconditional if `balance` is off) -- so a tree
//   that has fallen more than treeRatio behind keeps getting grown instead of losing
//   its turn to the tree that is already ahead. Because OMPL populates the goal tree
//   lazily inside the loop where VAMP seeds it before the loop starts, the population
//   step is ordered before the balance check here so the sizes it compares are never
//   stale relative to what VAMP would have seen.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/PlannerIncludes.h>
#include <ompl/tools/config/SelfConfig.h>

namespace ompl::geometric
{
    class RRTConnectDD : public base::Planner
    {
    public:
        explicit RRTConnectDD(const base::SpaceInformationPtr &si)
          : base::Planner(si, "RRTConnectDD")
        {
            specs_.recognizedGoal = base::GOAL_SAMPLEABLE_REGION;
            specs_.directed = true;
            Planner::declareParam<double>("range", this, &RRTConnectDD::setRange, &RRTConnectDD::getRange,
                                          "0.:1.:10000.");
            connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
            distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
        }

        ~RRTConnectDD() override
        {
            freeMemory();
        }

        void setRange(double distance)
        {
            maxDistance_ = distance;
        }

        double getRange() const
        {
            return maxDistance_;
        }

        /// vamp::planning::RRTCSettings::dynamic_domain. On by default, matching VAMP.
        void setDynamicDomain(bool enabled)
        {
            dynamicDomain_ = enabled;
        }

        bool getDynamicDomain() const
        {
            return dynamicDomain_;
        }

        /// RRTCSettings::radius / alpha / min_radius, in the state space's own units --
        /// both here and in VAMP that is radians, so VAMP's numeric defaults (4.0,
        /// 0.0001, 1.0) transfer directly with no conversion.
        void setDomainParameters(double radius, double alpha, double minRadius)
        {
            domainRadius_ = radius;
            domainAlpha_ = alpha;
            domainMinRadius_ = minRadius;
        }

        /// RRTCSettings::balance / tree_ratio. On by default, ratio 1.0, matching VAMP.
        void setBalanced(bool enabled, double treeRatio = 1.0)
        {
            balance_ = enabled;
            treeRatio_ = treeRatio;
        }

        bool getBalanced() const
        {
            return balance_;
        }

        /// When true, the connect phase matches rrtc.hh exactly: find the other tree's
        /// nearest node to the just-extended one ONCE, then walk *the just-extended
        /// tree* toward it in `ceil(d/range)` fixed increments -- the other tree is
        /// never touched, unlike stock RRTConnect's connect phase, which repeatedly
        /// re-queries nearest and grows *the other tree* instead. See growVampConnect().
        void setVampConnect(bool enabled)
        {
            vampConnect_ = enabled;
        }

        bool getVampConnect() const
        {
            return vampConnect_;
        }

        void setup() override
        {
            Planner::setup();
            tools::SelfConfig sc(si_, getName());
            sc.configurePlannerRange(maxDistance_);

            if (!tStart_)
                tStart_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
            if (!tGoal_)
                tGoal_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
            tStart_->setDistanceFunction([this](const Motion *a, const Motion *b)
                                         { return distanceFunction(a, b); });
            tGoal_->setDistanceFunction([this](const Motion *a, const Motion *b)
                                        { return distanceFunction(a, b); });
        }

        void clear() override
        {
            Planner::clear();
            sampler_.reset();
            freeMemory();
            if (tStart_)
                tStart_->clear();
            if (tGoal_)
                tGoal_->clear();
            connectionPoint_ = std::make_pair<base::State *, base::State *>(nullptr, nullptr);
            distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
            startTree_ = true;
        }

        base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override
        {
            checkValidity();
            auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());
            if (goal == nullptr)
            {
                OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
                return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
            }

            while (const base::State *st = pis_.nextStart())
            {
                auto *motion = new Motion(si_);
                si_->copyState(motion->state, st);
                motion->root = motion->state;
                tStart_->add(motion);
            }

            if (tStart_->size() == 0)
            {
                OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
                return base::PlannerStatus::INVALID_START;
            }
            if (!goal->couldSample())
            {
                OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
                return base::PlannerStatus::INVALID_GOAL;
            }
            if (!sampler_)
                sampler_ = si_->allocStateSampler();

            OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),
                        (int)(tStart_->size() + tGoal_->size()));

            TreeGrowingInfo tgi;
            tgi.xstate = si_->allocState();

            Motion *approxsol = nullptr;
            double approxdif = std::numeric_limits<double>::infinity();
            auto *rmotion = new Motion(si_);
            base::State *rstate = rmotion->state;
            bool solved = false;
            base::PlannerStatus::StatusType status = base::PlannerStatus::TIMEOUT;

            while (!ptc)
            {
                // Populate the goal tree before the balance check reads its size, so a
                // just-seeded goal is not read as "behind" the moment it appears --
                // VAMP has no equivalent step because it seeds the whole goal set
                // before the loop even starts.
                if (tGoal_->size() == 0 || pis_.getSampledGoalsCount() < tGoal_->size() / 2)
                {
                    const base::State *st = tGoal_->size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
                    if (st != nullptr)
                    {
                        auto *motion = new Motion(si_);
                        si_->copyState(motion->state, st);
                        motion->root = motion->state;
                        tGoal_->add(motion);
                    }
                    if (tGoal_->size() == 0)
                    {
                        OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
                        status = base::PlannerStatus::INVALID_GOAL;
                        break;
                    }
                }

                // `tree`/`otherTree` are bound as a fixed complementary pair for THIS
                // iteration, from startTree_'s value at loop entry -- exactly stock's
                // pre-toggle `tree` selection, with `otherTree` now always its
                // complement by construction rather than re-derived from startTree_ a
                // second time. The toggle below only ever decides what startTree_ will
                // select *next* iteration; it must never retroactively change who this
                // iteration's otherTree is, or a "don't toggle" balance decision makes
                // otherTree alias tree, silently merging both trees' growth into one.
                TreeData &tree = startTree_ ? tStart_ : tGoal_;
                TreeData &otherTree = startTree_ ? tGoal_ : tStart_;
                tgi.start = startTree_;
                const bool treeIsStart = tgi.start;  // `tree`'s identity, fixed before
                                                     // the toggle below -- vampConnect
                                                     // grows `tree`, so it needs this
                                                     // one, not the reassignment further
                                                     // down (which is otherTree's).
                {
                    const double asize = static_cast<double>(tree->size());
                    const double bsize = static_cast<double>(otherTree->size());
                    const double ratio = asize > 0.0 ? std::abs(asize - bsize) / asize : 0.0;
                    if (!balance_ || ratio < treeRatio_)
                        startTree_ = !startTree_;
                }

                sampler_->sampleUniform(rstate);

                GrowState gs = growTree(tree, tgi, rmotion, /*gateDynamicDomain=*/true);
                if (gs == TRAPPED)
                    continue;

                Motion *addedMotion = tgi.xmotion;
                if (gs != REACHED)
                    si_->copyState(rstate, tgi.xstate);
                tgi.start = startTree_;

                // `treeFar`/`otherFar` are which node ended up farthest on each side
                // this iteration -- always addedMotion/tgi.xmotion in stock mode
                // (otherTree grows), but in vampConnect mode `tree` keeps growing
                // instead and otherTree's node is the pre-existing nearest, untouched.
                Motion *treeFar = addedMotion;
                Motion *otherFar = nullptr;
                GrowState gsc;
                if (vampConnect_)
                {
                    gsc = growVampConnect(tree, addedMotion, otherTree, treeFar, otherFar, tgi.xstate, treeIsStart);
                }
                else
                {
                    gsc = growTree(otherTree, tgi, rmotion, /*gateDynamicDomain=*/false);
                    if (gsc == TRAPPED)
                        tgi.start = !tgi.start;
                    while (gsc == ADVANCED)
                        gsc = growTree(otherTree, tgi, rmotion, /*gateDynamicDomain=*/false);
                    otherFar = tgi.xmotion;
                }

                const double newDist = tree->getDistanceFunction()(addedMotion, otherTree->nearest(addedMotion));
                if (newDist < distanceBetweenTrees_)
                    distanceBetweenTrees_ = newDist;

                Motion *startMotion = tgi.start ? otherFar : treeFar;
                Motion *goalMotion = tgi.start ? treeFar : otherFar;

                if (gsc == REACHED && goal->isStartGoalPairValid(startMotion->root, goalMotion->root))
                {
                    if (startMotion->parent != nullptr)
                        startMotion = startMotion->parent;
                    else
                        goalMotion = goalMotion->parent;

                    connectionPoint_ = std::make_pair(startMotion->state, goalMotion->state);

                    Motion *solution = startMotion;
                    std::vector<Motion *> mpath1;
                    while (solution != nullptr)
                    {
                        mpath1.push_back(solution);
                        solution = solution->parent;
                    }
                    solution = goalMotion;
                    std::vector<Motion *> mpath2;
                    while (solution != nullptr)
                    {
                        mpath2.push_back(solution);
                        solution = solution->parent;
                    }

                    auto path(std::make_shared<PathGeometric>(si_));
                    path->getStates().reserve(mpath1.size() + mpath2.size());
                    for (int i = static_cast<int>(mpath1.size()) - 1; i >= 0; --i)
                        path->append(mpath1[i]->state);
                    for (auto &m : mpath2)
                        path->append(m->state);

                    pdef_->addSolutionPath(path, false, 0.0, getName());
                    solved = true;
                    break;
                }
                // Skipped entirely in vampConnect mode: tgi.xmotion there is stale (only
                // growTree's stock connect phase updates it), and VAMP's own connect
                // loop has no approximate-solution concept to port -- it either
                // connects or it doesn't, no partial credit.
                if (!vampConnect_ && tgi.start)
                {
                    double dist = 0.0;
                    goal->isSatisfied(tgi.xmotion->state, &dist);
                    if (dist < approxdif)
                    {
                        approxdif = dist;
                        approxsol = tgi.xmotion;
                    }
                }
            }

            si_->freeState(tgi.xstate);
            si_->freeState(rstate);
            delete rmotion;

            OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(),
                        tStart_->size() + tGoal_->size(), tStart_->size(), tGoal_->size());

            if (approxsol && !solved)
            {
                std::vector<Motion *> mpath;
                while (approxsol != nullptr)
                {
                    mpath.push_back(approxsol);
                    approxsol = approxsol->parent;
                }
                auto path(std::make_shared<PathGeometric>(si_));
                for (int i = static_cast<int>(mpath.size()) - 1; i >= 0; --i)
                    path->append(mpath[i]->state);
                pdef_->addSolutionPath(path, true, approxdif, getName());
                return base::PlannerStatus::APPROXIMATE_SOLUTION;
            }
            return solved ? base::PlannerStatus::EXACT_SOLUTION : status;
        }

        void getPlannerData(base::PlannerData &data) const override
        {
            Planner::getPlannerData(data);
            std::vector<Motion *> motions;
            if (tStart_)
                tStart_->list(motions);
            for (auto &motion : motions)
            {
                if (motion->parent == nullptr)
                    data.addStartVertex(base::PlannerDataVertex(motion->state, 1));
                else
                    data.addEdge(base::PlannerDataVertex(motion->parent->state, 1),
                                 base::PlannerDataVertex(motion->state, 1));
            }
            motions.clear();
            if (tGoal_)
                tGoal_->list(motions);
            for (auto &motion : motions)
            {
                if (motion->parent == nullptr)
                    data.addGoalVertex(base::PlannerDataVertex(motion->state, 2));
                else
                    data.addEdge(base::PlannerDataVertex(motion->state, 2),
                                 base::PlannerDataVertex(motion->parent->state, 2));
            }
            data.addEdge(data.vertexIndex(connectionPoint_.first), data.vertexIndex(connectionPoint_.second));
            data.properties["approx goal distance REAL"] = std::to_string(distanceBetweenTrees_);
        }

    protected:
        class Motion
        {
        public:
            Motion() = default;
            explicit Motion(const base::SpaceInformationPtr &si) : state(si->allocState())
            {
            }
            ~Motion() = default;

            const base::State *root{nullptr};
            base::State *state{nullptr};
            Motion *parent{nullptr};
            /// Dynamic-domain radius. Infinity means "never failed", matching VAMP's
            /// float-max sentinel exactly (see the class comment).
            double radius{std::numeric_limits<double>::infinity()};
        };

        using TreeData = std::shared_ptr<NearestNeighbors<Motion *>>;

        struct TreeGrowingInfo
        {
            base::State *xstate;
            Motion *xmotion;
            bool start;
        };

        enum GrowState
        {
            TRAPPED,
            ADVANCED,
            REACHED
        };

        void freeMemory()
        {
            std::vector<Motion *> motions;
            if (tStart_)
            {
                tStart_->list(motions);
                for (auto &m : motions)
                {
                    if (m->state != nullptr)
                        si_->freeState(m->state);
                    delete m;
                }
            }
            if (tGoal_)
            {
                motions.clear();
                tGoal_->list(motions);
                for (auto &m : motions)
                {
                    if (m->state != nullptr)
                        si_->freeState(m->state);
                    delete m;
                }
            }
        }

        double distanceFunction(const Motion *a, const Motion *b) const
        {
            return si_->distance(a->state, b->state);
        }

        /// \p gateDynamicDomain restricts the dynamic-domain radius check to the first
        /// (random-sample-directed) extension of an iteration, matching rrtc.hh: the
        /// connect-phase extensions that follow are never radius-gated.
        GrowState growTree(TreeData &tree, TreeGrowingInfo &tgi, Motion *rmotion, bool gateDynamicDomain)
        {
            Motion *nmotion = tree->nearest(rmotion);

            bool reach = true;
            base::State *dstate = rmotion->state;
            double d = si_->distance(nmotion->state, rmotion->state);

            if (gateDynamicDomain && dynamicDomain_ && nmotion->radius < d)
                return TRAPPED;  // untouched radius: nothing attempted, nothing learned

            if (d > maxDistance_)
            {
                si_->getStateSpace()->interpolate(nmotion->state, rmotion->state, maxDistance_ / d, tgi.xstate);
                if (si_->equalStates(nmotion->state, tgi.xstate))
                    return TRAPPED;
                dstate = tgi.xstate;
                reach = false;
            }

            bool validMotion = tgi.start ? si_->checkMotion(nmotion->state, dstate) :
                                           si_->isValid(dstate) && si_->checkMotion(dstate, nmotion->state);

            if (!validMotion)
            {
                if (dynamicDomain_)
                {
                    if (!std::isfinite(nmotion->radius))
                        nmotion->radius = domainRadius_;
                    else
                        nmotion->radius = std::max(nmotion->radius * (1.0 - domainAlpha_), domainMinRadius_);
                }
                return TRAPPED;
            }

            auto *motion = new Motion(si_);
            si_->copyState(motion->state, dstate);
            motion->parent = nmotion;
            motion->root = nmotion->root;
            tree->add(motion);
            tgi.xmotion = motion;

            if (dynamicDomain_ && std::isfinite(nmotion->radius))
                nmotion->radius *= (1.0 + domainAlpha_);

            return reach ? REACHED : ADVANCED;
        }

        /// rrtc.hh's connect phase, verbatim in spirit: find `otherTree`'s nearest node
        /// to `from` ONCE (returned via \p otherNode, otherTree itself untouched), then
        /// grow `tree` -- the tree `from` already belongs to, not otherTree -- toward it
        /// in `ceil(d / maxDistance_)` fixed increments computed from the two fixed
        /// endpoints (mathematically identical to accumulating a constant per-step
        /// delta, since interpolation here is linear). \p outLast receives the farthest
        /// motion actually added to `tree` (or `from` itself if already coincident).
        ///
        /// Returns REACHED only if every increment validated and the walk reached
        /// otherNode exactly -- rrtc.hh's own "connected" test (`i_extension ==
        /// n_extensions`) has no partial-credit notion, so a walk that gets partway is
        /// reported ADVANCED (some progress, for the approxsol path) rather than
        /// TRAPPED, and TRAPPED only when not even the first increment validates.
        GrowState growVampConnect(TreeData &tree, Motion *from, TreeData &otherTree, Motion *&outLast,
                                  Motion *&otherNode, base::State *scratch, bool treeIsStart)
        {
            Motion *nearest = otherTree->nearest(from);
            otherNode = nearest;
            outLast = from;

            const double d = si_->distance(from->state, nearest->state);
            if (d < 1e-9)
                return REACHED;

            const auto n = static_cast<std::size_t>(std::ceil(d / maxDistance_));
            Motion *prior = from;
            std::size_t i = 1;
            for (; i <= n; ++i)
            {
                si_->getStateSpace()->interpolate(from->state, nearest->state, static_cast<double>(i) / n, scratch);
                const bool validMotion = treeIsStart ? si_->checkMotion(prior->state, scratch) :
                                                       si_->isValid(scratch) && si_->checkMotion(scratch, prior->state);
                if (!validMotion)
                    break;

                auto *motion = new Motion(si_);
                si_->copyState(motion->state, scratch);
                motion->parent = prior;
                motion->root = prior->root;
                tree->add(motion);
                prior = motion;
            }
            outLast = prior;
            if (i > n)
                return REACHED;
            return (prior == from) ? TRAPPED : ADVANCED;
        }

        base::StateSamplerPtr sampler_;
        TreeData tStart_;
        TreeData tGoal_;
        bool startTree_{true};
        double maxDistance_{0.};
        RNG rng_;
        std::pair<base::State *, base::State *> connectionPoint_;
        double distanceBetweenTrees_;

        bool dynamicDomain_{true};
        double domainRadius_{4.0};
        double domainAlpha_{0.0001};
        double domainMinRadius_{1.0};
        bool balance_{true};
        double treeRatio_{1.0};
        bool vampConnect_{false};
    };
}  // namespace ompl::geometric
