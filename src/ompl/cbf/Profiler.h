#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ompl::cbf
{
    /// Whether ScopedTimer and FilterStats::record actually do anything. Checked
    /// once (the environment does not change mid-run) rather than on every call,
    /// so "optional" costs one cached bool read, not a getenv() per filter call.
    /// Set OMPL_CBF_PROFILE=1 to turn instrumentation on. It is opt-in because each
    /// sample takes a process-wide mutex and materially distorts this microsecond-scale
    /// hot path.
    inline bool profilingEnabled()
    {
        static const bool enabled = []
        {
            const char *v = std::getenv("OMPL_CBF_PROFILE");
            return v != nullptr && std::strcmp(v, "0") != 0;
        }();
        return enabled;
    }

    /// Wall-time accumulator for a handful of named sections in the CBF hot path.
    /// A benchmark-only tool: global, process-wide, and with mutex overhead per
    /// sample, so it is not meant to stay on for anything but profiling runs.
    class Profiler
    {
    public:
        struct Entry
        {
            double seconds{0.0};
            long calls{0};
        };

        static Profiler &instance()
        {
            static Profiler p;
            return p;
        }

        void add(const std::string &name, double seconds)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            Entry &e = entries_[name];
            e.seconds += seconds;
            e.calls += 1;
        }

        /// Prints one line per section, sorted by total time descending.
        void report(std::FILE *out = stderr) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<std::pair<std::string, Entry>> sorted(entries_.begin(), entries_.end());
            std::sort(sorted.begin(), sorted.end(),
                     [](const auto &a, const auto &b) { return a.second.seconds > b.second.seconds; });

            std::fprintf(out, "=== CBF profiler (wall time) ===\n");
            for (const auto &[name, e] : sorted)
                std::fprintf(out, "  %-20s total=%9.4fs  calls=%8ld  avg=%9.3fus\n", name.c_str(),
                             e.seconds, e.calls, e.calls ? e.seconds / e.calls * 1e6 : 0.0);
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_.clear();
        }

    private:
        mutable std::mutex mutex_;
        std::unordered_map<std::string, Entry> entries_;
    };

    /// RAII wall-clock timer: on destruction, adds the elapsed time to
    /// Profiler::instance() under \p name.
    class ScopedTimer
    {
    public:
        explicit ScopedTimer(const char *name) : name_(name), active_(profilingEnabled())
        {
            if (active_)
                start_ = std::chrono::steady_clock::now();
        }

        ~ScopedTimer()
        {
            if (!active_)
                return;
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
            Profiler::instance().add(name_, seconds);
        }

    private:
        const char *name_;
        bool active_;
        std::chrono::steady_clock::time_point start_;
    };

    enum class FilterOutcome
    {
        Unchanged,
        Filtered,
        Blocked
    };

    /// Per-call outcome and diagnostic counters for a control filter, across the
    /// whole process. Separate from Profiler (wall time) since these are counts
    /// and values, not durations -- issue 9's filter-outcome, screening/activity,
    /// QP-iteration and certified-duration statistics all land here.
    class FilterStats
    {
    public:
        struct Snapshot
        {
            std::size_t calls{0};
            std::size_t unchanged{0};
            std::size_t filtered{0};
            std::size_t blocked{0};
            std::size_t activeRowsSum{0};
            std::size_t zeroActiveRows{0};
            int activeRowsMax{0};
            std::ptrdiff_t solverIterationsSum{0};
            std::ptrdiff_t solverIterationsMax{0};
            double certifiedDurationSum{0.0};
            double certifiedDurationMin{std::numeric_limits<double>::infinity()};
            double certifiedDurationMax{0.0};
            /// The gain each call actually needed, against the cap it was allowed. A
            /// mean and a max would not settle the question this is asked to settle --
            /// the max is set by the handful of near-contact calls in the run, and the
            /// mean by the thousands in open space -- so the distribution is kept, in
            /// four log-spaced buckets per decade from 1e-3 to 1e4 /s. That is 28
            /// counters, which is cheaper than the branch that fills them.
            static constexpr int gainBuckets = 28;
            static constexpr double gainLogMin = -3.0;  ///< log10 of the first bucket edge
            static constexpr double gainPerDecade = 4.0;
            std::array<std::size_t, gainBuckets> gainHistogram{};
            double gainSum{0.0};
            double gainMax{0.0};
            std::size_t gainFinite{0};
            /// Calls whose region certified no gain at all -- a barrier already at or
            /// below zero, so no allowance makes the row hold. Counted, never averaged
            /// in: folding an infinity into a mean would destroy the mean and hide how
            /// rare it is.
            std::size_t gainInfinite{0};
        };

        static FilterStats &instance()
        {
            static FilterStats s;
            return s;
        }

        /// \p activeRows and \p solverIterations are always meaningful (screening
        /// runs, and produces a row count, even for a Blocked call); \p certifiedDuration
        /// and \p requiredGain are meaningful only when a control was actually returned,
        /// i.e. not Blocked. \p requiredGain defaults to infinity so a filter that does
        /// not compute one is recorded as certifying no gain rather than a gain of zero.
        void record(FilterOutcome outcome, int activeRows, std::ptrdiff_t solverIterations,
                   double certifiedDuration,
                   double requiredGain = std::numeric_limits<double>::infinity())
        {
            if (!profilingEnabled())
                return;
            std::lock_guard<std::mutex> lock(mutex_);
            Snapshot &s = snapshot_;
            ++s.calls;
            switch (outcome)
            {
                case FilterOutcome::Unchanged:
                    ++s.unchanged;
                    break;
                case FilterOutcome::Filtered:
                    ++s.filtered;
                    break;
                case FilterOutcome::Blocked:
                    ++s.blocked;
                    break;
            }
            s.activeRowsSum += static_cast<std::size_t>(activeRows);
            if (activeRows == 0)
                ++s.zeroActiveRows;
            s.activeRowsMax = std::max(s.activeRowsMax, activeRows);
            s.solverIterationsSum += solverIterations;
            s.solverIterationsMax = std::max(s.solverIterationsMax, solverIterations);
            if (outcome != FilterOutcome::Blocked)
            {
                s.certifiedDurationSum += certifiedDuration;
                s.certifiedDurationMin = std::min(s.certifiedDurationMin, certifiedDuration);
                s.certifiedDurationMax = std::max(s.certifiedDurationMax, certifiedDuration);

                if (std::isfinite(requiredGain))
                {
                    ++s.gainFinite;
                    s.gainSum += requiredGain;
                    s.gainMax = std::max(s.gainMax, requiredGain);
                    // A gain of zero -- a control that moves no barrier -- is real and
                    // belongs in the first bucket rather than at log10 of minus infinity.
                    const double decades =
                        requiredGain > 0.0
                            ? (std::log10(requiredGain) - Snapshot::gainLogMin) * Snapshot::gainPerDecade
                            : 0.0;
                    const int bucket = static_cast<int>(
                        std::clamp(decades, 0.0, static_cast<double>(Snapshot::gainBuckets - 1)));
                    ++s.gainHistogram[static_cast<std::size_t>(bucket)];
                }
                else
                {
                    ++s.gainInfinite;
                }
            }
        }

        void report(std::FILE *out = stderr) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const Snapshot &s = snapshot_;
            const auto pct = [&](std::size_t part)
            { return s.calls ? 100.0 * static_cast<double>(part) / static_cast<double>(s.calls) : 0.0; };

            std::fprintf(out, "=== CBF filter stats ===\n");
            std::fprintf(out, "  calls=%zu  unchanged=%zu (%.1f%%)  filtered=%zu (%.1f%%)  blocked=%zu (%.1f%%)\n",
                        s.calls, s.unchanged, pct(s.unchanged), s.filtered, pct(s.filtered), s.blocked,
                        pct(s.blocked));
            std::fprintf(out, "  active rows:       avg=%.2f  max=%d\n",
                        s.calls ? static_cast<double>(s.activeRowsSum) / static_cast<double>(s.calls) : 0.0,
                        s.activeRowsMax);
            std::fprintf(out, "  zero-row fast path: %zu (%.1f%%)\n", s.zeroActiveRows,
                        pct(s.zeroActiveRows));
            std::fprintf(out, "  qp iterations:     avg=%.2f  max=%td\n",
                        s.calls ? static_cast<double>(s.solverIterationsSum) / static_cast<double>(s.calls) : 0.0,
                        s.solverIterationsMax);
            const std::size_t certifiedCalls = s.unchanged + s.filtered;
            std::fprintf(out, "  certified duration: avg=%.5fs  min=%.5fs  max=%.5fs  (over %zu non-blocked calls)\n",
                        certifiedCalls ? s.certifiedDurationSum / static_cast<double>(certifiedCalls) : 0.0,
                        certifiedCalls ? s.certifiedDurationMin : 0.0, s.certifiedDurationMax, certifiedCalls);

            // The gain the run actually needed, against the cap it was given. Quantiles
            // come off the histogram's upper edges, so each is an upper bound on the true
            // one and reads high by at most a quarter decade -- which is the honest
            // direction for a number used to argue a cap could be lowered.
            std::fprintf(out, "  required gain:     avg=%.3f/s  max=%.3f/s  p50=%.3f/s  p90=%.3f/s  "
                              "p99=%.3f/s  (over %zu calls, %zu uncertified)\n",
                        s.gainFinite ? s.gainSum / static_cast<double>(s.gainFinite) : 0.0, s.gainMax,
                        gainQuantile(s, 0.50), gainQuantile(s, 0.90), gainQuantile(s, 0.99),
                        s.gainFinite, s.gainInfinite);
        }

        /// The \p fraction quantile of the required-gain histogram, as the upper edge of
        /// the bucket the fraction falls in. Infinite when that bucket is the last one,
        /// which is the overflow bin and so has no upper edge to report.
        static double gainQuantile(const Snapshot &s, double fraction)
        {
            if (s.gainFinite == 0)
                return 0.0;
            const auto target = static_cast<double>(s.gainFinite) * fraction;
            std::size_t seen = 0;
            for (int bucket = 0; bucket < Snapshot::gainBuckets; ++bucket)
            {
                seen += s.gainHistogram[static_cast<std::size_t>(bucket)];
                if (static_cast<double>(seen) >= target)
                {
                    if (bucket == Snapshot::gainBuckets - 1)
                        return std::numeric_limits<double>::infinity();
                    return std::pow(10.0, Snapshot::gainLogMin +
                                              (bucket + 1) / Snapshot::gainPerDecade);
                }
            }
            return std::numeric_limits<double>::infinity();
        }

        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_ = Snapshot();
        }

    private:
        mutable std::mutex mutex_;
        Snapshot snapshot_;
    };
}  // namespace ompl::cbf
