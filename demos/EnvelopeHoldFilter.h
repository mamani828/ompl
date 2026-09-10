#pragma once
#include "HoldTimeCertificate.h"

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/robots/UR5.h>

#include <algorithm>
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
            /// Sum of envelope span / L1 span, over calls where the L1 span was at
            /// least a hundredth of a step. Unconditioned, this ratio is dominated by
            /// calls whose L1 span is a rounding error, where any finite improvement
            /// reads as a factor of millions and says nothing.
            double gain{0.0};
            std::size_t gainCalls{0};
        };

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
            Filter::Diagnostics diagnostics;
            diagnostics.wantEvaluation = true;
            const Status status = inner_.filter(q, nominal, duration, filtered, diagnostics);
            certified = diagnostics.certifiedDuration;
            safe = status == Status::Blocked ? 0.0 : diagnostics.safeDuration;
            if (status == Status::Blocked || !parameters_.certificates ||
                diagnostics.evaluation == nullptr)
                return status;

            // The certificate is parameterized by the control itself, not by a unit
            // direction: every quantity in it is homogeneous in that parameterization,
            // so feeding it the joint velocity and a horizon in seconds returns seconds.
            const double speed = filtered.cwiseAbs().maxCoeff();
            if (!(speed > 0.0))
                return status;

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
            if (kGateEnabled && safe * gainCeiling_ < duration)
            {
                ++stats.gated;
                return status;
            }
            ++stats.calls;
            double horizon = std::max(kFloorSteps * duration, kGrowth * safe);
            double best = 0.0;
            for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
            {
                ++stats.queries;
                const std::size_t before = holdtime::emptyActiveCount();
                const double screenHorizon =
                    parameters_.kappa > 0.0
                        ? std::max(duration, 1.0 / parameters_.kappa)
                        : std::numeric_limits<double>::infinity();
                const double span =
                    kSubsetEnabled
                        ? engine_.holdScaleFrom(*diagnostics.evaluation, filtered, horizon,
                                                screenHorizon)
                        : engine_.holdScaleFrom(*diagnostics.evaluation, filtered, horizon);
                stats.emptyActive += holdtime::emptyActiveCount() - before;
                best = std::max(best, span);
                if (span < horizon * (1.0 - 1e-9))
                    break;   // the bound is interior: a longer horizon cannot help
                ++stats.saturated;
                horizon *= kGrowth;
            }

            // The QP's control box keeps `q + u dt` inside the joint limits, which says
            // nothing about a longer span. `CBFControlFilter` clamps its own spans for
            // this reason and the replacement has to be clamped the same way, or the hop
            // runs the arm out of travel.
            if (parameters_.respectJointLimits)
            {
                const Configuration lower = robots::UR5::lowerBounds();
                const Configuration upper = robots::UR5::upperBounds();
                for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(robots::UR5::nJoints); ++j)
                {
                    if (filtered[j] == 0.0)
                        continue;
                    const double room = (filtered[j] > 0.0 ? upper[j] : lower[j]) - q[j];
                    best = std::min(best, std::max(room / filtered[j], 0.0));
                }
            }

            if (safe > 0.01 * duration)
            {
                const double ratio = best / safe;
                stats.gain += ratio;
                ++stats.gainCalls;
                if (ratio > gainCeiling_)
                    gainCeiling_ = ratio;
            }
            stats.longer += best > safe ? 1u : 0u;
            stats.pastStep += std::max(best, safe) > duration ? 1u : 0u;
            stats.pastStepL1 += safe > duration ? 1u : 0u;
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
        /// Starting ceiling on envelope-span / L1-span, before any is measured.
        /// Generous: the measured mean is 4-10x, so this gates only calls that are
        /// nowhere near paying.
        mutable double gainCeiling_{32.0};

        static double kFloorSteps;
        static double kGrowth;
        static int kMaxAttempts;
        /// A/B switches for the two cost reductions, so one binary can run both arms.
        static bool kGateEnabled;
        static bool kSubsetEnabled;

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
}  // namespace ompl::demo
