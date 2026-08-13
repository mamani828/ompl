#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ompl::cbf
{
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
        explicit ScopedTimer(const char *name) : name_(name), start_(std::chrono::steady_clock::now())
        {
        }

        ~ScopedTimer()
        {
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
            Profiler::instance().add(name_, seconds);
        }

    private:
        const char *name_;
        std::chrono::steady_clock::time_point start_;
    };
}  // namespace ompl::cbf
