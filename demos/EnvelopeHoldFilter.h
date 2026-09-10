#pragma once
#include "HoldTimeCertificate.h"

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/robots/UR5.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace ompl::demo
{
    /// The CBF-QP filter with one number replaced: how long the control it solved for
    /// may be held.
    ///
    /// `CBFControlFilter` answers that from `ClearanceBarrier::durations()`, the L1
    /// lever-arm diamond restricted to the ray the control traces — every barrier's
    /// clearance divided by a configuration-independent bound on how fast that barrier
    /// can fall. This wraps the same filter, keeps its QP, its rows, its screening, its
    /// control and its `certified` span, and re-answers the `safe` span from the
    /// motion-envelope hold certificate instead.
    ///
    /// So a planner comparison between "QP plus L1 hop" and "QP plus envelope hop"
    /// differs by exactly one object. The rollout, the state space, the planner, the
    /// seeds and the barrier are the same; `FilteredStateSpace::roll()` is not touched.
    ///
    /// ### Why the span may be replaced without weakening anything
    ///
    /// Both spans certify the same statement — every `h_i >= 0` over the whole interval
    /// — off the same clearances, at the same configuration, for the same control. The
    /// envelope one is longer because it bounds the sphere centres' *motion* rather than
    /// their worst-case lever arms, and because each row takes the maximum with its own
    /// L1 time. Reporting `max` of the two is therefore both sound and never shorter
    /// than what the wrapped filter would have said on its own.
    ///
    /// ### The horizon, and why it is not a constant
    ///
    /// The hold certificate is stated over a finite horizon `T` and clips its answer
    /// there, and `T` is not free: the envelope widens the angular bounds over the whole
    /// of `[0, T]`, so asking about a horizon far longer than the hop makes the answer
    /// *shorter*, not longer. The rollout's usable hop is bounded by whatever is left of
    /// the edge, which a `ControlFilter` cannot see. So the horizon starts a few
    /// multiples of the L1 span out and grows only while the answer comes back pinned to
    /// it — each attempt is a valid certificate for its own horizon, so keeping the
    /// largest is sound, and a saturated answer is the only evidence that a longer look
    /// would have paid.
    class EnvelopeHoldFilter final : public cbf::ControlFilter
    {
    public:
        using Barrier = cbf::ClearanceBarrier;
        using Filter = cbf::CBFControlFilter;

        struct Stats
        {
            std::size_t calls{0};       ///< filter calls that reached the certificate
            std::size_t queries{0};     ///< hold queries issued, including horizon retries
            std::size_t saturated{0};   ///< queries that came back pinned to their horizon
            std::size_t longer{0};      ///< calls where the envelope span beat the L1 one
            std::size_t pastStep{0};    ///< calls whose span exceeded one controller step
            std::size_t pastStepL1{0};  ///< ...and how many the L1 span alone already did
            std::size_t gated{0};       ///< calls that skipped the query as unable to pay
            std::size_t emptyActive{0}; ///< queries answered with no geometry at all
            /// Queries re-asked unscreened because the screened answer was resting on
            /// the `max(dt, 1/kappa)` clip rather than on a constraint.
            std::size_t rescreened{0};

            /// Wall time, split so the two ways this filter can cost more than the one
            /// it wraps are separable: `inner` is the wrapped QP, which it pays exactly
            /// like `qpAdaptive` does, and `certificate` is everything this class adds
            /// on top -- the gate test on every call plus the query on the calls that
            /// pass it. Only filled under `OMPL_HOLD_TIMING`, because a clock read is a
            /// measurable share of a query this size.
            double innerSeconds{0.0};
            double certificateSeconds{0.0};
            /// ...of which, calls the gate turned away. Their cost is pure overhead: no
            /// certificate is produced, so anything here is a tax on the 87% that the
            /// gate exists to make cheap.
            double gatedSeconds{0.0};

            /// What stopped the envelope span, one bucket per call that reached the
            /// certificate. The question these answer is which geometry the hop is
            /// actually paying for: a hop capped by the workspace wants a finer field
            /// or a larger horizon, one capped by the arm against itself wants neither.
            /// Exactly one of the first three is incremented per call.
            std::size_t worldBound{0};   ///< a world sphere row set the span
            std::size_t selfBound{0};    ///< a self-collision pair row set it
            std::size_t unbound{0};      ///< neither: the horizon or the screening clip
            std::size_t blocked{0};      ///< the span came back zero, so no hop at all
            /// Calls where the joint-limit clamp cut the certified span below what the
            /// geometry rows allowed -- a bottleneck in the arm's travel, not in the
            /// scene. Counted on top of the bucket above, which names the row the clamp
            /// then overrode.
            std::size_t limitBound{0};

            /// The binding row's answer is `max(L1, speed-capped, anchored)`, and this
            /// is which of the three attained it, split by family because `anchored`
            /// means the normal-anchored root solve on a self pair and the weaker
            /// isotropic one on a world row. Indexed by `HoldTerm` less one, so
            /// `[0]` is L1, `[1]` speed-capped, `[2]` anchored.
            ///
            /// Ties go to the cheapest term, so `[0]` counts calls where the free L1
            /// time was already the row's whole answer and the dearer terms bought
            /// nothing. That is the return on the certificate, per binding row.
            ///
            /// Indexed by `HoldTerm` less one: `[0]` L1, `[1]` local-lever L1,
            /// `[2]` speed-capped, `[3]` anchored. The local term is world-only.
            ///
            /// These sum to `worldBound + selfBound` less `blocked`: a blocked row
            /// never forms the terms.
            std::size_t worldTerm[4]{};
            std::size_t selfTerm[4]{};

            /// The span the rollout is handed, in controller steps: `[0]` under one step,
            /// then 1-2, 2-4, 4-8, 8+. `l1Steps` is what it would have been given the
            /// wrapped filter's L1 answer alone, `envSteps` what it is given after the
            /// envelope. Only the movement of mass out of `[0]` can change a hop, because
            /// below one step the rollout's floor decides instead.
            ///
            /// Every call that reaches the certificate is counted, gated ones included --
            /// they are the bulk of the calls and they are exactly the ones the ratio
            /// statistic hides.
            static constexpr int buckets = 5;
            std::size_t l1Steps[buckets]{};
            std::size_t envSteps[buckets]{};
        };

        /// Which `Stats::l1Steps` bucket a span of \p seconds falls in, at a controller
        /// step of \p step.
        static int stepBucket(double seconds, double step)
        {
            if (!(step > 0.0) || seconds < step)
                return 0;
            const double steps = seconds / step;
            if (steps < 2.0)
                return 1;
            if (steps < 4.0)
                return 2;
            if (steps < 8.0)
                return 3;
            return 4;
        }

        /// Process-wide, because the benchmark builds one filter per problem and the
        /// question is about the method, not about any one scene.
        static Stats &aggregate()
        {
            static Stats s;
            return s;
        }

        EnvelopeHoldFilter(const Barrier &barrier, const Filter::Parameters &parameters)
          : barrier_(barrier), inner_(barrier, parameters), engine_(barrier),
            parameters_(parameters)
        {
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered) const override
        {
            return inner_.filter(q, nominal, duration, filtered);
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered, double &certified) const override
        {
            return inner_.filter(q, nominal, duration, filtered, certified);
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered, double &certified, double &safe) const override
        {
            // The evaluation the QP just made is borrowed rather than repeated: the
            // certificate needs the same forward kinematics, the same 40 sphere centres
            // and the same barrier values, and rebuilding them was most of what this
            // wrapper used to cost.
            const bool timing = holdtime::holdTimingEnabled();
            const auto t0 = timing ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
            Filter::Diagnostics diagnostics;
            diagnostics.wantEvaluation = true;
            const Status status = inner_.filter(q, nominal, duration, filtered, diagnostics);
            const auto t1 = timing ? std::chrono::steady_clock::now() : t0;
            if (timing)
                aggregate().innerSeconds += std::chrono::duration<double>(t1 - t0).count();
            /// Charges everything after the wrapped filter to the certificate, however
            /// this call leaves.
            struct Charge
            {
                bool on;
                std::chrono::steady_clock::time_point start;
                double *bucket;
                ~Charge()
                {
                    if (!on)
                        return;
                    const double dt = std::chrono::duration<double>(
                                          std::chrono::steady_clock::now() - start)
                                          .count();
                    aggregate().certificateSeconds += dt;
                    if (bucket != nullptr)
                        *bucket += dt;
                }
            } charge{timing, t1, nullptr};
            certified = diagnostics.certifiedDuration;
            safe = status == Status::Blocked ? 0.0 : diagnostics.safeDuration;
            if (status == Status::Blocked || !parameters_.certificates ||
                diagnostics.evaluation == nullptr)
                return status;

            // The certificate is parameterized by the control itself, not by a unit
            // direction: every quantity in it is homogeneous in that parameterization,
            // so feeding it the joint velocity and a horizon in seconds returns seconds.
            Stats &stats = aggregate();

            // The rollout floors its hop at one controller step -- the QP already
            // certified that much -- so a certificate that cannot reach past a step
            // changes nothing and is not worth computing. The L1 span is in hand for
            // free, and the envelope span is an empirical multiple of it, so a call
            // whose L1 span is far below a step is skipped.
            //
            // The ceiling is the largest ratio seen so far, floored at a generous
            // constant and only ever revised upward, so the gate loosens as evidence
            // arrives and never tightens on it. And skipping is a speed decision with
            // no safety content: the fallback is the L1 span, which is exactly what the
            // wrapped filter would have reported on its own.
            // `gainCeiling_` only ever rises, to the largest ratio ever seen, so the gate
            // admits anything that could pay at the best case ever recorded. The typical
            // ratio is a small single digit, so a cap here trades a little conversion for
            // a lot of queries; `OMPL_ENV_CEILING` sweeps that trade.
            // Tested before anything else this class computes. It rejects ~98% of calls
            // at one multiply and one compare, so every operation ordered above it is
            // paid fifty times over for the two per cent that get past. `aggregate()` is
            // a function-local static, hence a guard load per call: `stats` is the same
            // reference and is already in hand.
            if (kGateEnabled && safe * kGateRatio < duration)
            {
                ++stats.gated;
                charge.bucket = &stats.gatedSeconds;
                const int bucket = stepBucket(safe, duration);
                ++stats.l1Steps[bucket];
                ++stats.envSteps[bucket];
                return status;
            }

            // Only the query needs this, and a zero control would make the certificate
            // meaningless -- every quantity in it is homogeneous in the control, so a
            // zero one gives no span to scale. Ordered after the gate so the 98% never
            // pay for it. A zero-control call whose L1 span is already sub-step now
            // counts as gated rather than as neither, which is the one accounting
            // difference the reorder makes.
            const double speed = filtered.cwiseAbs().maxCoeff();
            if (!(speed > 0.0))
                return status;
            ++stats.calls;
            double horizon = std::max(kFloorSteps * duration, kGrowth * safe);
            double best = 0.0;
            // Attribution follows the attempt that produced `best`. Spans are
            // non-decreasing in the horizon, so that is the last attempt run, but the
            // comparison is written out rather than assumed.
            holdtime::HoldEngine::Report report, chosen;
            report.wantSelfL1 = false;   // the L1 sweep is a diagnostic this does not read
            for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
            {
                ++stats.queries;
                const std::size_t before = holdtime::emptyActiveCount();
                // Must track `CBFControlFilter`'s own screen exactly: the clip is only
                // sound out to the horizon the dropped rows were proved against -- and
                // with screening off no row is dropped, so there is nothing to clip
                // against and the horizon is unbounded. Reading `kappa` alone would clip
                // an unscreened evaluation for a screen that never ran.
                const double screenHorizon =
                    (parameters_.screening && parameters_.kappa > 0.0)
                        ? std::max(1.0, parameters_.screenHorizonScale) *
                              std::max(duration, 1.0 / parameters_.kappa)
                        : std::numeric_limits<double>::infinity();
                double span =
                    kSubsetEnabled
                        ? engine_.holdScaleFrom(*diagnostics.evaluation, filtered, horizon,
                                                screenHorizon, &report)
                        : engine_.holdScaleFrom(*diagnostics.evaluation, filtered, horizon,
                                                &report);

                // The screened query clips its answer at `screenHorizon` -- the span the
                // QP's own active-set screen was done at, `max(dt, 1/kappa)`. That clip
                // was justified as "far longer than any hop"; at kappa 20 and a 10 ms
                // step it is five steps, and the longest hops sit on it.
                //
                // An answer resting on the clip is the *subset* running out of validity,
                // not the certificate running out of clearance, and the loop below cannot
                // lift it: growing the horizon leaves the screen exactly where it was. So
                // re-ask over all 303 pairs, where there is no clip. This is the only
                // escalation that can help, and it fires only when the clip is what
                // stopped the answer, which is a few percent of calls.
                if (kEscalateEnabled && kSubsetEnabled && screenHorizon < horizon &&
                    span >= screenHorizon * (1.0 - 1e-9))
                {
                    ++stats.queries;
                    ++stats.rescreened;
                    span = engine_.holdScaleFrom(*diagnostics.evaluation, filtered, horizon,
                                                 &report);
                }
                stats.emptyActive += holdtime::emptyActiveCount() - before;
                if (span >= best)
                    chosen = report;
                best = std::max(best, span);
                if (span < horizon * (1.0 - 1e-9))
                    break;   // the bound is interior: a longer horizon cannot help
                ++stats.saturated;
                horizon *= kGrowth;
            }

            if (!chosen.bounded)
                ++stats.unbound;
            else if (chosen.selfBinding)
                ++stats.selfBound;
            else
                ++stats.worldBound;
            stats.blocked += chosen.blocked ? 1u : 0u;
            if (chosen.bindingTerm != holdtime::HoldTerm::None)
            {
                const int term = static_cast<int>(chosen.bindingTerm) - 1;
                ++(chosen.selfBinding ? stats.selfTerm : stats.worldTerm)[term];
            }

            // The QP's control box keeps `q + u dt` inside the joint limits, which says
            // nothing about a longer span. `CBFControlFilter` clamps its own spans for
            // this reason and the replacement has to be clamped the same way, or the hop
            // runs the arm out of travel.
            if (parameters_.respectJointLimits)
            {
                const Configuration lower = robots::UR5::lowerBounds();
                const Configuration upper = robots::UR5::upperBounds();
                const double certified = best;
                for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(robots::UR5::nJoints); ++j)
                {
                    if (filtered[j] == 0.0)
                        continue;
                    const double room = (filtered[j] > 0.0 ? upper[j] : lower[j]) - q[j];
                    best = std::min(best, std::max(room / filtered[j], 0.0));
                }
                stats.limitBound += best < certified ? 1u : 0u;
            }

            stats.longer += best > safe ? 1u : 0u;
            stats.pastStep += std::max(best, safe) > duration ? 1u : 0u;
            stats.pastStepL1 += safe > duration ? 1u : 0u;
            // Bucketed after the joint-limit clamp and against `max`, so both rows are
            // the span the rollout is actually handed, not the certificate in isolation.
            ++stats.l1Steps[stepBucket(safe, duration)];
            ++stats.envSteps[stepBucket(std::max(safe, best), duration)];
            safe = std::max(safe, best);
            return status;
        }

        const char *name() const override
        {
            return "cbf-qp + envelope hold";
        }


    private:
        /// The horizon never starts below this many controller steps, so a call whose L1
        /// span is tiny still gets a look worth taking.

        static double kFloorSteps;
        static double kGrowth;
        static int kMaxAttempts;
        /// A/B switches for the two cost reductions, so one binary can run both arms.
        static bool kGateEnabled;
        static bool kSubsetEnabled;
        static bool kEscalateEnabled;
        /// Gate threshold: a call is queried only when its L1 span, multiplied by this,
        /// could reach one controller step. Replaces an adaptive ceiling that rose to the
        /// largest ratio ever seen -- which made the gate *looser* the more good luck it
        /// saw, the opposite of what it wanted -- and which the measured cap of 2 rendered
        /// inert anyway. See the sweep in the results README.
        static double kGateRatio;

        const Barrier &barrier_;
        Filter inner_;
        mutable holdtime::HoldEngine engine_;
        Filter::Parameters parameters_;
    };

    // Horizon policy, overridable from the environment while it is being tuned.
    inline double envelopeEnv(const char *name, double fallback)
    {
        const char *v = std::getenv(name);
        return v != nullptr ? std::atof(v) : fallback;
    }
    inline double EnvelopeHoldFilter::kFloorSteps = envelopeEnv("OMPL_ENV_FLOOR", 4.0);
    inline double EnvelopeHoldFilter::kGrowth = envelopeEnv("OMPL_ENV_GROWTH", 4.0);
    inline int EnvelopeHoldFilter::kMaxAttempts =
        static_cast<int>(envelopeEnv("OMPL_ENV_ATTEMPTS", 3.0));
    inline bool EnvelopeHoldFilter::kGateEnabled = envelopeEnv("OMPL_ENV_GATE", 1.0) != 0.0;
    inline bool EnvelopeHoldFilter::kSubsetEnabled = envelopeEnv("OMPL_ENV_SUBSET", 1.0) != 0.0;
    inline bool EnvelopeHoldFilter::kEscalateEnabled =
        envelopeEnv("OMPL_ENV_ESCALATE", 1.0) != 0.0;
    /// Two, measured. `gainCeiling_` rises to the largest span ratio ever seen -- tens
    /// -- and the gate then admits any call that could reach a controller step at that
    /// best case. Sweeping the cap on the MotionBenchMaker set at 10 ms and 50 ms steps,
    /// every value from 2 upward yields the *same* barrier-evaluation count, while the
    /// query count and therefore the certificate's overhead keep climbing: at a 10 ms
    /// step, 1.4% of the wrapped filter's time at 2 against 2.9% uncapped, for an
    /// identical 4.66% reduction in evaluations. Below 2 the reduction starts to go
    /// (4.55% at 1.5), so this is the corner.
    inline double EnvelopeHoldFilter::kGateRatio = envelopeEnv("OMPL_ENV_CEILING", 2.0);
}  // namespace ompl::demo
