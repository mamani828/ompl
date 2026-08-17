#pragma once

#include <algorithm>
#include <chrono>
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
        };

        static FilterStats &instance()
        {
            static FilterStats s;
            return s;
        }

        /// \p activeRows and \p solverIterations are always meaningful (screening
        /// runs, and produces a row count, even for a Blocked call); \p certifiedDuration
        /// is meaningful only when a control was actually returned, i.e. not Blocked.
        void record(FilterOutcome outcome, int activeRows, std::ptrdiff_t solverIterations,
                   double certifiedDuration)
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
