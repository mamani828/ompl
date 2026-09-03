#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/ConfigurationOperations.h>
#include <ompl/cbf/ControlFilter.h>
#include <ompl/util/Exception.h>

namespace ompl::cbf
{
    /// A joint-space state space whose `interpolate()` is a CBF rollout instead of a
    /// straight line, and which keeps the trajectory it produced.
    ///
    /// This is the geometric counterpart to `FilteredStatePropagator`, and it exists
    /// because control-space planning turned out to be the wrong shape for speed. A
    /// control edge is capped at `maxControlDuration * stepSize * maxSpeed` of joint
    /// travel, so a wide motion needs many short edges and the tree grows deep:
    /// measured at ~110-770 vertices where `geometric::RRTConnect` needs 5 on the same
    /// scene. Putting the rollout behind `interpolate()` lets a *geometric* planner
    /// take long-range extensions while every intermediate state is still certified by
    /// the barrier.
    ///
    /// ### The edge is the trajectory
    ///
    /// A tree edge in stock OMPL is two endpoints; the motion between them is whatever
    /// the state space says it is, recomputed on demand. That does not work here. The
    /// nominal control re-aims at its target every step, so a rollout aimed at the
    /// deflected endpoint `x` is a *different motion* from the one that produced `x`
    /// while aiming at the original sample. Recomputing gives a plausible, safe, wrong
    /// answer -- one that passes every barrier check while not being the trajectory the
    /// planner actually found.
    ///
    /// So the rollout is run once and its waypoints are kept, keyed by the edge they
    /// produced. Every later question about that edge -- is it valid, what does it look
    /// like at fraction t, what should the robot execute -- is answered from that
    /// record. Nothing is ever re-derived. `FilteredMotionValidator` accepts an edge by
    /// looking it up rather than by re-rolling it, which is where the cost went.
    ///
    /// This is exact, not an approximation: the rollout integrates a control held
    /// constant over each step, so the straight line between two consecutive waypoints
    /// *is* the executed motion. A densified record is therefore a genuinely geometric
    /// path, and sampling strictly inside a step is meaningful rather than invented.
    ///
    /// ### Straight where it can be, filtered where it must be
    ///
    /// A rollout does not step at a fixed rate. Each filter call also hands back how
    /// long the control it returned stays certified -- by default the span over which no
    /// barrier can reach zero, and with `setSafeHops(false)` the shorter span over which
    /// nothing the filter enforces can bind -- and the rollout runs that control for
    /// exactly that long before asking again. `setSafeHops()` documents what separates
    /// the two and what the longer one costs. Where there is room, one call certifies the whole
    /// extension and the edge is a single straight line, produced without checking
    /// anything along it, because there is nothing along it left to find out. Where
    /// there is not, the certificate is short, the step falls back to `stepSize`, and
    /// the QP does the work as before. Nothing in between is guessed at: the two
    /// regimes are separated by a Lipschitz bound over the interval, not by a
    /// heuristic, so the coarse steps are the *more* trustworthy of the two -- they
    /// hold at every point of the span, where a QP step holds at its endpoints and
    /// leans on the margin in between.
    ///
    /// This is why the certificate is worth having at all: an extension through open
    /// space used to cost one QP solve per 0.025 rad and now costs one evaluation
    /// total. `setMaxStepScale(1)` puts the fixed step back for comparison.
    ///
    /// ### The contract, and where it bends
    ///
    /// OMPL asks that `interpolate(a, b, 1, out)` give `out == b`. That holds here only
    /// when the rollout is unobstructed -- and when it is, this space *is* a
    /// `RealVectorStateSpace`, up to quantisation: the nominal control re-aims at `b`
    /// every step, so in free space the rollout advances exactly `|b - a| / N` per step
    /// and lands on `b`. Near an obstacle the rollout slides along the boundary and
    /// ends somewhere else, which is the entire point: the deflected endpoint is a
    /// perfectly good new tree node, and it is safe, whereas a straight line into the
    /// obstacle is not.
    ///
    /// A planner that stores `interpolate()`'s output as a node is therefore storing a
    /// state the rollout actually produced, and this space has that rollout on file.
    /// `geometric::RRT` does exactly that on its long-extension branch
    /// (`d > maxDistance_`). On the short branch it stores the raw sample instead,
    /// which the rollout may not reach -- that case is what `FilteredMotionValidator`'s
    /// arrive-or-reject fallback is for.
    ///
    /// ### What it does not give you
    ///
    /// `distance()` is still Euclidean, and is now a *lower bound* on the length of
    /// the motion `interpolate()` would actually take, since sliding around an
    /// obstacle is longer than cutting through it. That only makes it a slightly
    /// looser nearest-neighbour heuristic, which is all a tree planner needs it for.
    ///
    /// Not thread safe: the ledger and the statistics are mutable, and `ControlFilter`
    /// already cost thread safety anyway since `CBFControlFilter` solves into shared
    /// scratch.
    template <typename Robot>
    class RobotFilteredStateSpace : public base::RealVectorStateSpace
    {
    public:
        using Filter = RobotControlFilter<Robot>;
        using Configuration = typename Robot::Configuration;
        using Control = typename Robot::Configuration;
        using Operations = RobotConfigurationOperations<Robot>;

        static constexpr int dimension = static_cast<int>(Robot::nJoints);
        static_assert(Configuration::RowsAtCompileTime == dimension,
                      "Robot::Configuration must have Robot::nJoints rows");
        static_assert(Configuration::ColsAtCompileTime == 1,
                      "Robot::Configuration must be a column vector");

        struct Rollout
        {
            Configuration end;                     ///< where the rollout finished
            std::vector<Configuration> waypoints;  ///< the executed motion; front() is the start
            unsigned int steps{0};                 ///< filter calls made
            unsigned int filtered{0};              ///< of those, how many the CBF altered
            unsigned int blocked{0};               ///< of those, how many had no safe control
            unsigned int coarse{0};                ///< of those, how many ran past `stepSize`
            double travel{0.0};                    ///< joint-space radians covered
            double fraction{0.0};                  ///< share of the full horizon it got through
            bool reachedTarget{false};             ///< did it finish within reachTolerance of `to`?
            bool callBudgetReached{false};         ///< stopped after the configured filter-call budget
            bool stalled{false};                   ///< consecutive steps made negligible target progress
            bool tinyControl{false};               ///< stopped on a numerically negligible applied control
        };

        /// Optional sequential-rollout work limits. Disabled by default because `roll()`
        /// is also a public integration primitive; planners opt in when a useful partial
        /// edge is preferable to spending an unbounded number of calls on one target.
        struct EarlyTermination
        {
            bool enabled{false};
            /// Zero leaves the number of filter calls unbounded.
            /// Forty is deliberately above a typical useful extension: shorter caps
            /// fragmented RRTConnect's greedy connect chain and increased total work.
            unsigned int maxFilterCalls{40};
            /// Stop after this many consecutive low-progress steps; zero disables it.
            unsigned int stalledSteps{3};
            /// A step is stalled when target-distance reduction is at most this share
            /// of the nominal step's travel.
            double minProjectedProgressFraction{0.01};
            /// Applied max joint-speed fraction below which the control is treated as zero.
            double minControlFraction{1e-4};
        };

        /// Optional speculative steering implementation. Returning false asks the space
        /// to run its ordinary sequential rollout, so experiments can be installed
        /// without weakening the production fallback. A rejected proposal may return
        /// filter-work counters in its Rollout; they are charged but its motion is not.
        using RolloutPlanner =
            std::function<bool(const Configuration &, const Configuration &, double, Rollout &)>;

        /// Aggregate counters, so a planner run can be costed without instrumenting
        /// the planner. Mutable because `interpolate()` is const.
        struct Statistics
        {
            std::size_t rollouts{0};
            std::size_t steps{0};  ///< filter calls actually made
            std::size_t filtered{0};
            std::size_t blocked{0};
            std::size_t coarse{0};  ///< steps that ran past `stepSize` on a certificate
            double travel{0.0};     ///< joint-space radians rolled; `travel / steps` is what a
                                    ///< filter call buys, which is the number to quote a cost at
            std::size_t abandoned{0};  ///< rollouts discarded for making no progress
            std::size_t served{0};     ///< queries answered from the ledger, at no filter cost
            std::size_t recorded{0};   ///< edges committed to the ledger
            std::size_t evicted{0};    ///< edges dropped for capacity; should stay zero
            std::size_t proposalAttempts{0};
            std::size_t proposalAccepted{0};
            std::size_t proposalFallbacks{0};
            std::size_t callBudgetTerminations{0};
            std::size_t stallTerminations{0};
            std::size_t tinyControlTerminations{0};
        };

        /// \p filter is not copied and must outlive this space. \p stepSize is the
        /// rollout integration step -- the CBF's linearisation is only valid over a
        /// small step, so this is a correctness parameter, not a performance knob.
        RobotFilteredStateSpace(const Filter &filter, double stepSize)
          : RobotFilteredStateSpace(filter, stepSize, Robot::velocityLimits())
        {
        }

        /// As above, with custom per-joint speed limits. State bounds still come from
        /// `Robot::lowerBounds()` and `Robot::upperBounds()`.
        RobotFilteredStateSpace(const Filter &filter, double stepSize, const Control &maxSpeed)
          : RobotFilteredStateSpace(filter, stepSize, maxSpeed, Robot::lowerBounds(),
                                    Robot::upperBounds())
        {
        }

        /// As above, with explicit runtime bounds (notably a mobile base workspace).
        RobotFilteredStateSpace(const Filter &filter, double stepSize, const Control &maxSpeed,
                                const Configuration &lower, const Configuration &upper)
          : base::RealVectorStateSpace(dimension), filter_(filter), stepSize_(stepSize),
            maxSpeed_(maxSpeed)
        {
            if (stepSize <= 0.0)
                throw Exception("FilteredStateSpace: stepSize must be positive");
            if ((maxSpeed.array() <= 0.0).any())
                throw Exception("FilteredStateSpace: every maxSpeed entry must be positive");
            base::RealVectorBounds bounds(static_cast<unsigned int>(dimension));
            for (int j = 0; j < dimension; ++j)
            {
                bounds.setLow(static_cast<unsigned int>(j), lower[j]);
                bounds.setHigh(static_cast<unsigned int>(j), upper[j]);
            }
            setBounds(bounds);
            setName("Filtered" + getName());
        }

        /// How close a rollout must finish to `to` for it to count as having got there.
        ///
        /// Only the arrive-or-reject fallback in `FilteredMotionValidator` uses this --
        /// a recorded edge is answered by lookup and needs no tolerance. Exact equality
        /// is the right answer in free space but useless near an obstacle, where two
        /// rollouts from the same state toward slightly different targets deflect
        /// slightly differently. Defaults to one full-speed step.
        double reachTolerance() const
        {
            return reachTolerance_ > 0.0 ? reachTolerance_
                                         : Operations::defaultReachTolerance(maxSpeed_, stepSize_);
        }

        void setReachTolerance(double tolerance)
        {
            reachTolerance_ = tolerance;
        }

        /// The number of steps a full (t = 1) rollout from \p from to \p to takes: the
        /// time the straight-line motion would need at the per-joint speed limit,
        /// divided by the step size. At least one whenever the states differ.
        unsigned int horizonSteps(const Configuration &from, const Configuration &to) const
        {
            const double horizon = Operations::duration(from, to, maxSpeed_);
            return static_cast<unsigned int>(std::ceil(horizon / stepSize_ - 1e-12));
        }

        /// Roll out from \p from toward \p to for \p fraction of the full horizon,
        /// keeping every state it passes through.
        ///
        /// The nominal control re-aims at \p to at every step rather than being held
        /// constant. Both choices reduce to a straight line in free space, but
        /// re-aiming also recovers the original intent after the filter deflects it,
        /// which is what makes a long extension useful rather than merely safe.
        ///
        /// A pure function of (`from`, `to`, `fraction`): no memo, no resumption. What
        /// used to be a prefix cache is now the ledger, which keeps whole edges rather
        /// than one trajectory prefix, and is consulted by `interpolate()` before this
        /// is ever called.
        Rollout roll(const Configuration &from, const Configuration &to, double fraction) const
        {
            const unsigned int total = horizonSteps(from, to);
            const double horizon = static_cast<double>(total) * stepSize_;
            const double budget = std::clamp(fraction, 0.0, 1.0) * horizon;

            // Durations below this are indistinguishable from having arrived, and joint
            // displacements below the second are indistinguishable from zero: at ~1e-12
            // rad even the longest lever arm on the arm moves by a picometre.
            const double negligibleTime = 1e-9 * stepSize_;
            constexpr double negligibleAngle = 1e-12;

            Rollout out;
            out.end = from;
            // Deliberately not `total + 1`: a certified edge holds two waypoints and the
            // vector is moved into the ledger keeping whatever it reserved.
            out.waypoints.reserve(std::min<std::size_t>(total + 1, 16));
            out.waypoints.push_back(from);

            bool terminal = false;
            Control nominal;
            Control applied;
            double elapsed = 0.0;
            unsigned int consecutiveStalls = 0;
            while (budget - elapsed > negligibleTime)
            {
                // Time left in the *full* horizon, so a truncated rollout follows the
                // same trajectory as the prefix of a complete one.
                const double remaining = horizon - elapsed;
                nominal = Operations::difference(out.end, to) / remaining;
                for (int j = 0; j < dimension; ++j)
                    nominal[j] = std::clamp(nominal[j], -maxSpeed_[j], maxSpeed_[j]);

                double certified = 0.0;
                double safe = 0.0;
                const typename Filter::Status status =
                    filter_.filter(out.end, nominal, stepSize_, applied, certified, safe);
                // Which of the two certificates the hop is allowed to spend. See
                // `setSafeHops()` for what the longer one gives up.
                const double reach = safeHops_ ? safe : certified;
                if (status == Filter::Status::Blocked)
                {
                    // Nothing safe to do. Stop rather than sit still burning steps --
                    // the caller gets a short rollout and can decide.
                    terminal = true;
                    break;
                }
                if (status == Filter::Status::Filtered)
                    ++out.filtered;

                ++out.steps;

                // A QP can return a nonzero control whose motion is numerically useless.
                // Treat it as zero before integrating a long chain of microscopic states.
                if (earlyTermination_.enabled && earlyTermination_.minControlFraction > 0.0)
                {
                    const double appliedFraction =
                        applied.cwiseAbs().cwiseQuotient(maxSpeed_).maxCoeff();
                    if (appliedFraction <= earlyTermination_.minControlFraction)
                    {
                        out.tinyControl = true;
                        break;
                    }
                }

                // How far to run what the filter just handed back. The floor is the step
                // it was asked about, which it answered for; above that the filter has
                // certified itself a no-op, so running on is not an extrapolation but a
                // saving of calls whose outcome is already known.
                const double span =
                    std::min(std::max(stepSize_, std::min(reach, maxStepScale_ * stepSize_)),
                             budget - elapsed);

                const Configuration previous = out.end;
                const double previousGap = Operations::distance(previous, to, maxSpeed_);
                Configuration landing = Operations::integrate(previous, applied, span);
                // A hop that runs the horizon out was aimed to finish on `to`, and with
                // nothing in the way it does -- to the last bit. Recognising that lets an
                // unobstructed edge end on the state that was asked for rather than one
                // rounding away from it, which is what the ledger keys on.
                if (Operations::difference(landing, to).cwiseAbs().maxCoeff() <= negligibleAngle)
                    landing = Operations::normalize(to);

                elapsed += span;
                // With early termination disabled, a cornered-but-feasible zero control
                // runs the clock out without adding repeated waypoints. The enabled path
                // catches exact and numerical zero controls above before integration.
                const bool moved = !bitwiseEqual(landing, previous);

                if (moved)
                {
                    out.travel += Operations::distance(previous, landing, maxSpeed_);
                    out.end = landing;
                    out.waypoints.push_back(out.end);
                    if (span > stepSize_)
                        ++out.coarse;
                }

                if (earlyTermination_.enabled && earlyTermination_.stalledSteps > 0)
                {
                    const Configuration nominalLanding =
                        Operations::integrate(previous, nominal, span);
                    const double nominalTravel =
                        Operations::distance(previous, nominalLanding, maxSpeed_);
                    const double progress =
                        previousGap - Operations::distance(out.end, to, maxSpeed_);
                    if (nominalTravel > negligibleAngle &&
                        progress <= earlyTermination_.minProjectedProgressFraction * nominalTravel)
                        ++consecutiveStalls;
                    else
                        consecutiveStalls = 0;

                    if (consecutiveStalls >= earlyTermination_.stalledSteps &&
                        budget - elapsed > negligibleTime)
                    {
                        out.stalled = true;
                        break;
                    }
                }

                if (earlyTermination_.enabled && earlyTermination_.maxFilterCalls > 0 &&
                    out.steps >= earlyTermination_.maxFilterCalls &&
                    budget - elapsed > negligibleTime)
                {
                    out.callBudgetReached = true;
                    break;
                }
            }

            if (terminal)
                out.blocked = 1;
            out.fraction = horizon > 0.0 ? elapsed / horizon : 0.0;
            out.reachedTarget =
                budget - elapsed <= negligibleTime &&
                Operations::distance(out.end, to, maxSpeed_) <= reachTolerance();

            account(out);
            return out;
        }

        Rollout roll(const base::State *from, const base::State *to, double fraction) const
        {
            return roll(configurationOf(from), configurationOf(to), fraction);
        }

        /// Use the installed speculative planner when it can certify a proposal, and
        /// otherwise preserve the exact direct-rollout behavior.
        Rollout steer(const Configuration &from, const Configuration &to, double fraction) const
        {
            if (rolloutPlanner_)
            {
                ++statistics_.proposalAttempts;
                Rollout proposal;
                if (rolloutPlanner_(from, to, fraction, proposal))
                {
                    ++statistics_.proposalAccepted;
                    account(proposal);
                    return proposal;
                }
                ++statistics_.proposalFallbacks;
                accountRejectedWork(proposal);
            }
            return roll(from, to, fraction);
        }

        Rollout steer(const base::State *from, const base::State *to, double fraction) const
        {
            return steer(configurationOf(from), configurationOf(to), fraction);
        }

        void setRolloutPlanner(RolloutPlanner planner)
        {
            rolloutPlanner_ = std::move(planner);
        }

        void clearRolloutPlanner()
        {
            rolloutPlanner_ = RolloutPlanner();
        }

        /// A recorded edge, oriented the way it was asked for.
        ///
        /// Valid until the next `record()`, which may rehash or evict. Both call sites
        /// consume it immediately.
        class EdgeRecord
        {
        public:
            EdgeRecord() = default;
            EdgeRecord(const std::vector<Configuration> *waypoints, bool reversed)
              : waypoints_(waypoints), reversed_(reversed)
            {
            }

            explicit operator bool() const
            {
                return waypoints_ != nullptr;
            }

            /// True when this edge was found stored the other way round. Safe to use:
            /// a recorded polyline visits the same states and the same segments in
            /// either direction. It is *not* a rollout run backwards -- the filter is
            /// not invertible -- and it does not need to be, because what is being
            /// asserted is the geometry of the motion, not the control that made it.
            bool reversed() const
            {
                return reversed_;
            }

            std::size_t size() const
            {
                return waypoints_ == nullptr ? 0 : waypoints_->size();
            }

            /// The i-th waypoint in *query* order.
            const Configuration &operator[](std::size_t i) const
            {
                return reversed_ ? (*waypoints_)[waypoints_->size() - 1 - i] : (*waypoints_)[i];
            }

            /// The state at fraction \p t of the edge, linear between waypoints.
            ///
            /// Exact, including strictly inside a step: the control is held constant
            /// over a step, so the segment joining two consecutive waypoints is the
            /// executed motion rather than a stand-in for it. `t = 0` and `t = 1`
            /// return the endpoints bit-exactly.
            Configuration at(double t) const
            {
                const std::size_t count = size();
                if (count == 1)
                    return (*this)[0];

                const double u = std::clamp(t, 0.0, 1.0) * static_cast<double>(count - 1);
                const auto index = static_cast<std::size_t>(u);
                if (index + 1 >= count)
                    return (*this)[count - 1];
                const double fraction = u - static_cast<double>(index);
                return Operations::interpolate((*this)[index], (*this)[index + 1], fraction);
            }

        private:
            const std::vector<Configuration> *waypoints_{nullptr};
            bool reversed_{false};
        };

        /// The trajectory recorded for the edge \p from -> \p to, in either direction,
        /// or a false record if there is none.
        EdgeRecord recordedEdge(const Configuration &from, const Configuration &to) const
        {
            auto found = ledger_.find(Edge{from, to});
            if (found != ledger_.end())
            {
                ++statistics_.served;
                return EdgeRecord(&found->second, false);
            }
            found = ledger_.find(Edge{to, from});
            if (found != ledger_.end())
            {
                ++statistics_.served;
                return EdgeRecord(&found->second, true);
            }
            return EdgeRecord();
        }

        EdgeRecord recordedEdge(const base::State *from, const base::State *to) const
        {
            return recordedEdge(configurationOf(from), configurationOf(to));
        }

        /// Keep \p waypoints as the executed motion of the edge \p from -> \p to.
        ///
        /// Refuses degenerate edges and never overwrites: re-recording a key with a
        /// different polyline would make replay ambiguous, and the first record is the
        /// one the planner built its tree on.
        void record(const Configuration &from, const Configuration &to,
                    std::vector<Configuration> waypoints) const
        {
            if (waypoints.size() < 2 || bitwiseEqual(from, to))
                return;

            const Edge key{from, to};
            if (ledger_.find(key) != ledger_.end())
                return;

            ledgerWaypoints_ += waypoints.size();
            order_.push_back(key);
            ledger_.emplace(key, std::move(waypoints));
            ++statistics_.recorded;
            evictToCapacity();
        }

        /// Does the pending rollout describe the edge \p from -> \p to, either way round?
        ///
        /// `interpolate()` leaves its rollout here rather than in the ledger, because at
        /// that point the planner has not yet decided whether to keep the edge. The
        /// motion validator is the one that knows, and it commits.
        bool staged(const Configuration &from, const Configuration &to) const
        {
            return staged_.valid && ((bitwiseEqual(staged_.from, from) && bitwiseEqual(staged_.to, to)) ||
                                     (bitwiseEqual(staged_.from, to) && bitwiseEqual(staged_.to, from)));
        }

        bool staged(const base::State *from, const base::State *to) const
        {
            return staged(configurationOf(from), configurationOf(to));
        }

        /// Move the pending rollout into the ledger.
        ///
        /// Counts as served: the caller asked about an edge and got an answer that cost
        /// no filter calls. This is the common case during planning -- the validator is
        /// asking about the extension `interpolate()` just produced -- so leaving it out
        /// would make `statistics().served` read zero on a run that never re-rolled
        /// anything.
        void commitStaged() const
        {
            if (staged_.valid)
            {
                record(staged_.from, staged_.to, std::move(staged_.waypoints));
                ++statistics_.served;
            }
            staged_.waypoints.clear();
            staged_.valid = false;
        }

        /// Ledger size limit, in waypoints. Reaching it evicts oldest-first, which
        /// degrades an affected edge back to being re-derived rather than replayed --
        /// so `statistics().evicted` staying zero is part of the contract, not a
        /// nicety. One waypoint is 48 bytes; the default is roughly 50 MB.
        std::size_t ledgerCapacity() const
        {
            return ledgerCapacity_;
        }

        void setLedgerCapacity(std::size_t maxWaypoints)
        {
            ledgerCapacity_ = maxWaypoints;
            evictToCapacity();
        }

        std::size_t ledgerWaypoints() const
        {
            return ledgerWaypoints_;
        }

        std::size_t ledgerEdges() const
        {
            return ledger_.size();
        }

        void clearLedger() const
        {
            ledger_.clear();
            order_.clear();
            ledgerWaypoints_ = 0;
            staged_.waypoints.clear();
            staged_.valid = false;
        }

        /// Longest step the rollout may take, as a multiple of `stepSize`. Unbounded by
        /// default; `1.0` restores the fixed step exactly, which is the A/B.
        ///
        /// A step only exceeds `stepSize` when the filter has certified that it is a
        /// no-op over the whole of it (`ControlFilter::filter()`'s five-argument form),
        /// so the cap buys nothing in safety -- it exists to isolate the effect when
        /// measuring, and to keep waypoint spacing bounded for a consumer that wants it.
        double maxStepScale() const
        {
            return maxStepScale_;
        }

        /// Whether a hop may spend the *safety* certificate rather than the no-op one.
        ///
        /// On by default. A hop then runs until a barrier could reach zero rather than
        /// until a constraint row could bind -- longer by `1/kappa` on every row, and
        /// longer again because the certified region's slack is the whole clearance
        /// rather than the clearance less that lookahead.
        ///
        /// What that buys is filter calls, which are most of the rollout's cost. On the
        /// MotionBenchMaker set it is 5-21% fewer barrier evaluations and 9-21% less
        /// planning time, the gain growing as `kappa` falls because that is what widens
        /// the gap between the two certificates.
        ///
        /// What it gives up is that an edge is no longer *the* filtered edge. It is
        /// collision-free, and the safe set is still forward invariant -- a hop ends with
        /// `h >= 0`, the filter resumes, and a zero control is always admissible for a
        /// single integrator, so no hop can strand the arm -- but inside a hop `h` may
        /// decay faster than the CBF's exponential envelope allows.
        ///
        /// Turn it off to recover the previous behaviour, where every edge is exactly the
        /// motion repeated filtering would have produced. That is what a caller wants if
        /// it intends to reproduce the trajectory a continuously-filtered execution would
        /// follow, rather than merely to plan a valid one.
        ///
        /// See `ClearanceBarrier::safeDuration()` against `certifiedDuration()`, and
        /// `ClearanceBarrier::noOpTraversalTime()` for the third option: keeping the
        /// envelope by slowing the traversal down instead of shortening the hop.
        bool safeHops() const
        {
            return safeHops_;
        }

        void setSafeHops(bool enabled)
        {
            safeHops_ = enabled;
        }

        void setMaxStepScale(double scale)
        {
            if (scale < 1.0)
                throw Exception("FilteredStateSpace: maxStepScale must be at least 1");
            maxStepScale_ = scale;
        }

        const EarlyTermination &earlyTermination() const
        {
            return earlyTermination_;
        }

        void setEarlyTermination(const EarlyTermination &parameters)
        {
            if (parameters.minProjectedProgressFraction < 0.0 ||
                parameters.minProjectedProgressFraction > 1.0)
                throw Exception("FilteredStateSpace: min projected progress must be in [0, 1]");
            if (parameters.minControlFraction < 0.0 || parameters.minControlFraction > 1.0)
                throw Exception("FilteredStateSpace: min control fraction must be in [0, 1]");
            earlyTermination_ = parameters;
        }

        /// Minimum share of the free-space progress an extension must actually achieve
        /// to be reported at all. See `interpolate()`.
        double minProgressFraction() const
        {
            return minProgressFraction_;
        }

        void setMinProgressFraction(double fraction)
        {
            minProgressFraction_ = fraction;
        }

        void interpolate(const base::State *from, const base::State *to, double t,
                         base::State *state) const override
        {
            // from may alias state, so finish reading before writing.
            const Configuration a = configurationOf(from);
            const Configuration b = configurationOf(to);

            // A recorded edge is a reconstruction, never an extension: replay it and
            // skip both the progress test (which only guards the creation of new edges)
            // and enforceBounds (every waypoint was in bounds when it was recorded).
            if (const EdgeRecord record = recordedEdge(a, b))
            {
                setState(state, record.at(t));
                return;
            }

            Rollout rollout = steer(a, b, t);
            staged_.waypoints.clear();
            staged_.valid = false;

            // An extension that makes no headway toward its target is not an extension,
            // and must be reported as going nowhere rather than as going sideways.
            //
            // This is not a nicety. `RRTConnect::growTree` returns ADVANCED whenever a
            // shortened extension is valid, and its connect loop
            // (`RRTConnect.cpp:294`, `while (gsc == ADVANCED)`) has no
            // termination-condition guard: it assumes each ADVANCED closes the gap by
            // `maxDistance_`, which is true of a straight line and false of a rollout
            // that slides along an obstacle boundary. Without this test that loop spins
            // forever and `solve()` never looks at the clock again -- observed as a hang
            // rather than a timeout. It matters more now, not less: the goal tree
            // accepts far more extensions than it used to.
            //
            // Note the threshold is relative to the progress a *free-space* rollout
            // would have made for this `t`, not absolute.
            const double span = Operations::distance(a, b, maxSpeed_);
            const double target = span * std::clamp(t, 0.0, 1.0);
            const double achieved = span - Operations::distance(rollout.end, b, maxSpeed_);
            if (achieved < minProgressFraction_ * target)
            {
                ++statistics_.abandoned;
                setState(state, a);
                enforceBounds(state);
                return;
            }

            setState(state, rollout.end);
            enforceBounds(state);

            // The recorded endpoint has to be bit-identical to the state the planner
            // will hand back, or the lookup misses forever. Clamping essentially never
            // happens -- the filter already respects the joint limits the bounds are
            // set from -- and when it does, the clamp displacement is not something the
            // filter certified, so the honest move is to not record and let the edge be
            // re-derived the old way.
            const Configuration stored = configurationOf(state);
            if (bitwiseEqual(stored, rollout.end))
            {
                staged_.from = a;
                staged_.to = stored;
                staged_.waypoints = std::move(rollout.waypoints);
                staged_.valid = true;
            }
        }

        double distance(const base::State *state1, const base::State *state2) const override
        {
            return Operations::distance(configurationOf(state1), configurationOf(state2), maxSpeed_);
        }

        void enforceBounds(base::State *state) const override
        {
            setState(state, Operations::normalize(configurationOf(state)));
            base::RealVectorStateSpace::enforceBounds(state);
        }

        static Configuration configurationOf(const base::State *state)
        {
            const double *values = state->as<StateType>()->values;
            Configuration q;
            for (int j = 0; j < dimension; ++j)
                q[j] = values[j];
            return q;
        }

        static void setState(base::State *state, const Configuration &q)
        {
            for (int j = 0; j < dimension; ++j)
                state->as<StateType>()->values[j] = q[j];
        }

        /// Bitwise equality, because the point is to recognise a state the planner just
        /// handed back unmodified -- `copyState` is a memcpy, so the bits survive. Not
        /// `operator==`, which calls -0.0 and 0.0 equal while their bit patterns differ;
        /// hashing and equality have to agree. A near-miss is a miss, which only costs
        /// a rollout.
        static bool bitwiseEqual(const Configuration &a, const Configuration &b)
        {
            return std::memcmp(a.data(), b.data(), sizeof(double) * dimension) == 0;
        }

        const Filter &filter() const
        {
            return filter_;
        }

        double stepSize() const
        {
            return stepSize_;
        }

        const Control &maxSpeed() const
        {
            return maxSpeed_;
        }

        const Statistics &statistics() const
        {
            return statistics_;
        }

        void resetStatistics() const
        {
            statistics_ = Statistics();
        }

    private:
        struct Edge
        {
            Configuration from;
            Configuration to;
        };

        struct EdgeHash
        {
            std::size_t operator()(const Edge &edge) const
            {
                return hashConfiguration(edge.to, hashConfiguration(edge.from, 0xcbf29ce484222325ull));
            }
        };

        struct EdgeEqual
        {
            bool operator()(const Edge &a, const Edge &b) const
            {
                return bitwiseEqual(a.from, b.from) && bitwiseEqual(a.to, b.to);
            }
        };

        /// The rollout `interpolate()` last produced, held until the motion validator
        /// says whether the planner is keeping the edge. Extensions that are rejected
        /// therefore cost nothing in memory.
        struct Staged
        {
            Configuration from{Configuration::Zero()};
            Configuration to{Configuration::Zero()};
            std::vector<Configuration> waypoints;
            bool valid{false};
        };

        static std::size_t hashConfiguration(const Configuration &q, std::uint64_t seed)
        {
            std::uint64_t hash = seed;
            for (int j = 0; j < dimension; ++j)
            {
                std::uint64_t word = 0;
                std::memcpy(&word, q.data() + j, sizeof(word));
                hash = (hash ^ word) * 0x100000001b3ull;
            }
            return static_cast<std::size_t>(hash);
        }

        void evictToCapacity() const
        {
            while (ledgerWaypoints_ > ledgerCapacity_ && !order_.empty())
            {
                const auto oldest = ledger_.find(order_.front());
                if (oldest != ledger_.end())
                {
                    ledgerWaypoints_ -= oldest->second.size();
                    ledger_.erase(oldest);
                    ++statistics_.evicted;
                }
                order_.pop_front();
            }
        }

        void account(const Rollout &rollout) const
        {
            ++statistics_.rollouts;
            // A terminal blocked call consumes filter work but integrates no step, so
            // Rollout::steps excludes it while the aggregate work counter includes it.
            statistics_.steps += rollout.steps + rollout.blocked;
            statistics_.filtered += rollout.filtered;
            statistics_.blocked += rollout.blocked;
            statistics_.coarse += rollout.coarse;
            statistics_.travel += rollout.travel;
            statistics_.callBudgetTerminations += rollout.callBudgetReached ? 1u : 0u;
            statistics_.stallTerminations += rollout.stalled ? 1u : 0u;
            statistics_.tinyControlTerminations += rollout.tinyControl ? 1u : 0u;
        }

        void accountRejectedWork(const Rollout &rollout) const
        {
            statistics_.steps += rollout.steps + rollout.blocked;
            statistics_.filtered += rollout.filtered;
            statistics_.blocked += rollout.blocked;
        }

        const Filter &filter_;
        double stepSize_;
        Control maxSpeed_;
        double reachTolerance_{-1.0};
        double maxStepScale_{std::numeric_limits<double>::infinity()};
        bool safeHops_{true};
        double minProgressFraction_{0.25};
        EarlyTermination earlyTermination_;
        RolloutPlanner rolloutPlanner_;
        std::size_t ledgerCapacity_{1u << 20};
        mutable Statistics statistics_;
        mutable Staged staged_;
        mutable std::unordered_map<Edge, std::vector<Configuration>, EdgeHash, EdgeEqual> ledger_;
        mutable std::deque<Edge> order_;  ///< insertion order, for eviction
        mutable std::size_t ledgerWaypoints_{0};
    };

    /// Construct a filtered state space directly from a robot model. The robot supplies
    /// its joint count, configuration type, limits, and default velocity limits; callers
    /// only supply the matching filter and integration step.
    template <typename Robot>
    std::shared_ptr<RobotFilteredStateSpace<Robot>> makeRobotFilteredStateSpace(
        const RobotControlFilter<Robot> &filter, double stepSize,
        const typename Robot::Configuration &maxSpeed = Robot::velocityLimits())
    {
        return std::make_shared<RobotFilteredStateSpace<Robot>>(filter, stepSize, maxSpeed);
    }

    /// Compatibility alias for the original UR5 API.
    using FilteredStateSpace = RobotFilteredStateSpace<robots::UR5>;
}  // namespace ompl::cbf
