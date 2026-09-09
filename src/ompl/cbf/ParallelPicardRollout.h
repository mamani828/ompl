#pragma once

#include <cstddef>
#include <memory>

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/cbf/FilteredStateSpace.h>

namespace ompl::cbf
{
    /// Speculative fixed-step CBF rollout using Picard iterations over short windows.
    ///
    /// Each Picard map evaluates the pointwise filter at every guessed state in a
    /// window concurrently, then prefix-integrates the returned controls. A final
    /// concurrent pass asks the ordinary CBF filter to accept every proposed segment
    /// at its actual start state. Only a fully accepted trajectory is returned;
    /// failure asks FilteredStateSpace to use its sequential rollout unchanged.
    class ParallelPicardRollout
    {
    public:
        using Configuration = robots::UR5::Configuration;
        using Control = Configuration;
        using Rollout = FilteredStateSpace::Rollout;

        struct Parameters
        {
            /// Number of fixed integration steps evaluated in one parallel window.
            unsigned int windowSteps{8};
            /// Picard map applications before the independent verification pass.
            unsigned int maxIterations{2};
            /// Under-relaxation of each map update, in (0, 1].
            double relaxation{1.0};
            /// Report a window as converged below this max joint-space residual.
            double convergenceTolerance{1e-3};
            /// Maximum control difference accepted by the final pointwise verification.
            double verificationTolerance{1e-10};
            /// Avoid parallel scheduling when fewer fixed steps remain.
            unsigned int minimumSteps{4};
            /// Zero selects min(windowSteps, hardware_concurrency), at least one.
            unsigned int workers{0};
        };

        struct Statistics
        {
            std::size_t attempts{0};
            std::size_t accepted{0};
            std::size_t fallbacks{0};
            std::size_t directCertificates{0};
            std::size_t windows{0};
            std::size_t convergedWindows{0};
            std::size_t mapBatches{0};
            std::size_t mapFilterCalls{0};
            /// Windows whose final Picard map exactly matched its input states, so
            /// those map calls themselves certify the emitted segments.
            std::size_t mapCertifiedWindows{0};
            std::size_t verificationBatches{0};
            std::size_t verificationFilterCalls{0};
            std::size_t verificationFailures{0};
            double maxResidual{0.0};
            double seconds{0.0};
        };

        explicit ParallelPicardRollout(const CBFControlFilter &filter);
        ParallelPicardRollout(const CBFControlFilter &filter, const Parameters &parameters);
        ~ParallelPicardRollout();

        ParallelPicardRollout(const ParallelPicardRollout &) = delete;
        ParallelPicardRollout &operator=(const ParallelPicardRollout &) = delete;

        /// Propose and independently certify a rollout. Returns false without a usable
        /// trajectory when the ordinary sequential fallback should be used; filter-work
        /// counters remain populated so rejected speculation is still accounted for.
        bool plan(const Configuration &from, const Configuration &to, double fraction,
                  double stepSize, const Control &maxSpeed, double maxStepScale,
                  double reachTolerance, Rollout &out);

        const Parameters &parameters() const;
        const Statistics &statistics() const;
        void resetStatistics();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace ompl::cbf
