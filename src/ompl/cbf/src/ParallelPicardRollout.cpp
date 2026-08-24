#include "ompl/cbf/ParallelPicardRollout.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <ompl/cbf/ConfigurationOperations.h>
#include <ompl/util/Exception.h>

namespace
{
    using Filter = ompl::cbf::CBFControlFilter;
    using Configuration = ompl::robots::UR5::Configuration;
    using Operations = ompl::cbf::RobotConfigurationOperations<ompl::robots::UR5>;

    class FilterBatch
    {
    public:
        using Task = std::function<void(std::size_t, Filter &)>;

        FilterBatch(const Filter &prototype, unsigned int requested)
        {
            const unsigned int count = std::max(1u, requested);
            filters_.reserve(count);
            threads_.reserve(count);
            for (unsigned int i = 0; i < count; ++i)
                filters_.push_back(
                    std::make_unique<Filter>(prototype.barrier(), prototype.parameters()));
            for (unsigned int i = 0; i < count; ++i)
                threads_.emplace_back([this, i] { worker(i); });
        }

        ~FilterBatch()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (std::thread &thread : threads_)
                thread.join();
        }

        FilterBatch(const FilterBatch &) = delete;
        FilterBatch &operator=(const FilterBatch &) = delete;

        void run(std::size_t count, Task task)
        {
            if (count == 0)
                return;

            std::unique_lock<std::mutex> lock(mutex_);
            count_ = count;
            task_ = std::move(task);
            next_.store(0, std::memory_order_relaxed);
            remaining_.store(threads_.size(), std::memory_order_relaxed);
            const std::size_t generation = ++generation_;
            ready_.notify_all();
            finished_.wait(lock, [this, generation] { return completedGeneration_ == generation; });
        }

    private:
        void worker(std::size_t workerIndex)
        {
            std::size_t observedGeneration = 0;
            while (true)
            {
                Task task;
                std::size_t count = 0;
                std::size_t generation = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [this, observedGeneration]
                                { return stopping_ || generation_ != observedGeneration; });
                    if (stopping_)
                        return;
                    task = task_;
                    count = count_;
                    generation = generation_;
                }

                while (true)
                {
                    const std::size_t index = next_.fetch_add(1, std::memory_order_relaxed);
                    if (index >= count)
                        break;
                    task(index, *filters_[workerIndex]);
                }

                if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    completedGeneration_ = generation;
                    finished_.notify_one();
                }
                observedGeneration = generation;
            }
        }

        std::vector<std::unique_ptr<Filter>> filters_;
        std::vector<std::thread> threads_;
        mutable std::mutex mutex_;
        std::condition_variable ready_;
        std::condition_variable finished_;
        Task task_;
        std::size_t count_{0};
        std::size_t generation_{0};
        std::size_t completedGeneration_{0};
        std::atomic<std::size_t> next_{0};
        std::atomic<std::size_t> remaining_{0};
        bool stopping_{false};
    };

    unsigned int defaultWorkers(unsigned int windowSteps)
    {
        const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
        return std::max(1u, std::min(windowSteps, hardware));
    }

    void clampControl(Configuration &u, const Configuration &maxSpeed)
    {
        for (Eigen::Index j = 0; j < u.size(); ++j)
            u[j] = std::clamp(u[j], -maxSpeed[j], maxSpeed[j]);
    }

    bool finite(const Configuration &q)
    {
        return q.allFinite();
    }

    ompl::cbf::ParallelPicardRollout::Parameters validateParameters(
        ompl::cbf::ParallelPicardRollout::Parameters parameters)
    {
        if (parameters.windowSteps == 0)
            throw ompl::Exception("ParallelPicardRollout: windowSteps must be positive");
        if (parameters.maxIterations == 0)
            throw ompl::Exception("ParallelPicardRollout: maxIterations must be positive");
        if (!(parameters.relaxation > 0.0 && parameters.relaxation <= 1.0))
            throw ompl::Exception("ParallelPicardRollout: relaxation must be in (0, 1]");
        if (parameters.convergenceTolerance <= 0.0 ||
            parameters.verificationTolerance < 0.0)
            throw ompl::Exception("ParallelPicardRollout: tolerances must be non-negative");
        if (parameters.minimumSteps == 0)
            throw ompl::Exception("ParallelPicardRollout: minimumSteps must be positive");
        return parameters;
    }
}  // namespace

struct ompl::cbf::ParallelPicardRollout::Impl
{
    Impl(const CBFControlFilter &prototype, Parameters p)
      : parameters(std::move(p))
      , probeFilter(prototype.barrier(), prototype.parameters())
      , batch(prototype, parameters.workers == 0 ? defaultWorkers(parameters.windowSteps)
                                                 : parameters.workers)
    {
    }

    Parameters parameters;
    Statistics statistics;
    CBFControlFilter probeFilter;
    FilterBatch batch;
};

ompl::cbf::ParallelPicardRollout::ParallelPicardRollout(const CBFControlFilter &filter)
  : ParallelPicardRollout(filter, Parameters())
{
}

ompl::cbf::ParallelPicardRollout::ParallelPicardRollout(const CBFControlFilter &filter,
                                                        const Parameters &parameters)
  : impl_(std::make_unique<Impl>(filter, validateParameters(parameters)))
{
}

ompl::cbf::ParallelPicardRollout::~ParallelPicardRollout() = default;

const ompl::cbf::ParallelPicardRollout::Parameters &
ompl::cbf::ParallelPicardRollout::parameters() const
{
    return impl_->parameters;
}

const ompl::cbf::ParallelPicardRollout::Statistics &
ompl::cbf::ParallelPicardRollout::statistics() const
{
    return impl_->statistics;
}

void ompl::cbf::ParallelPicardRollout::resetStatistics()
{
    impl_->statistics = Statistics();
}

bool ompl::cbf::ParallelPicardRollout::plan(const Configuration &from,
                                            const Configuration &to, double fraction,
                                            double stepSize, const Control &maxSpeed,
                                            double maxStepScale, double reachTolerance,
                                            Rollout &out)
{
    const auto begin = std::chrono::steady_clock::now();
    ++impl_->statistics.attempts;
    out = Rollout();

    std::size_t filterCalls = 0;
    std::size_t filteredCalls = 0;
    std::size_t blockedCalls = 0;

    const auto fail = [&]()
    {
        ++impl_->statistics.fallbacks;
        impl_->statistics.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        out = Rollout();
        out.steps = static_cast<unsigned int>(filterCalls - blockedCalls);
        out.filtered = static_cast<unsigned int>(filteredCalls);
        out.blocked = static_cast<unsigned int>(blockedCalls);
        return false;
    };

    if (!(stepSize > 0.0) || !std::isfinite(stepSize) || maxStepScale < 1.0 ||
        !finite(from) || !finite(to))
        return fail();

    const double minimumDuration = Operations::duration(from, to, maxSpeed);
    if (!(minimumDuration > 0.0) || !std::isfinite(minimumDuration))
        return fail();

    const unsigned int totalSteps =
        static_cast<unsigned int>(std::ceil(minimumDuration / stepSize - 1e-12));
    const double horizon = static_cast<double>(totalSteps) * stepSize;
    const double budget = std::clamp(fraction, 0.0, 1.0) * horizon;
    const unsigned int requestedSteps =
        static_cast<unsigned int>(std::ceil(budget / stepSize - 1e-12));
    if (requestedSteps < impl_->parameters.minimumSteps)
        return fail();

    // Preserve the production fast path: when one ordinary filter call certifies the
    // complete extension, Picard has no sequential chain left to parallelize.
    Control initialNominal = Operations::difference(from, to) / horizon;
    clampControl(initialNominal, maxSpeed);
    Control initialApplied;
    double initialCertified = 0.0;
    const CBFControlFilter::Status initialStatus =
        impl_->probeFilter.filter(from, initialNominal, stepSize, initialApplied,
                                  initialCertified);
    ++filterCalls;
    filteredCalls += initialStatus == CBFControlFilter::Status::Filtered ? 1u : 0u;
    blockedCalls += initialStatus == CBFControlFilter::Status::Blocked ? 1u : 0u;
    if (initialStatus == CBFControlFilter::Status::Blocked)
        return fail();

    const double directCertificate =
        std::max(stepSize, std::min(initialCertified, maxStepScale * stepSize));
    if (directCertificate + 1e-12 >= budget)
    {
        out.end = Operations::integrate(from, initialApplied, budget);
        if (Operations::difference(out.end, to).cwiseAbs().maxCoeff() <= 1e-12)
            out.end = Operations::normalize(to);
        out.waypoints = {from, out.end};
        out.steps = 1;
        out.filtered = static_cast<unsigned int>(filteredCalls);
        out.coarse = budget > stepSize ? 1u : 0u;
        out.travel = Operations::distance(from, out.end, maxSpeed);
        out.fraction = budget / horizon;
        out.reachedTarget = Operations::distance(out.end, to, maxSpeed) <= reachTolerance;
        ++impl_->statistics.accepted;
        ++impl_->statistics.directCertificates;
        impl_->statistics.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        return true;
    }

    out.end = from;
    out.waypoints.push_back(from);
    double elapsed = 0.0;
    bool firstMap = true;

    while (budget - elapsed > 1e-9 * stepSize)
    {
        const double windowBudget =
            std::min(static_cast<double>(impl_->parameters.windowSteps) * stepSize,
                     budget - elapsed);
        const unsigned int count =
            static_cast<unsigned int>(std::ceil(windowBudget / stepSize - 1e-12));
        if (count == 0)
            break;

        std::vector<double> spans(count, stepSize);
        spans.back() = windowBudget - static_cast<double>(count - 1) * stepSize;

        std::vector<Configuration> guess(count + 1);
        guess[0] = out.end;
        double localElapsed = 0.0;
        for (unsigned int k = 0; k < count; ++k)
        {
            Control nominal = Operations::difference(guess[k], to) /
                              (horizon - elapsed - localElapsed);
            clampControl(nominal, maxSpeed);
            guess[k + 1] = Operations::integrate(guess[k], nominal, spans[k]);
            localElapsed += spans[k];
        }

        std::vector<Configuration> controls(count);
        std::vector<CBFControlFilter::Status> statuses(
            count, CBFControlFilter::Status::Blocked);
        std::vector<Configuration> candidate(count + 1);
        double residual = std::numeric_limits<double>::infinity();
        bool converged = false;

        for (unsigned int iteration = 0; iteration < impl_->parameters.maxIterations;
             ++iteration)
        {
            std::vector<Configuration> nominals(count);
            localElapsed = 0.0;
            for (unsigned int k = 0; k < count; ++k)
            {
                nominals[k] = Operations::difference(guess[k], to) /
                              (horizon - elapsed - localElapsed);
                clampControl(nominals[k], maxSpeed);
                localElapsed += spans[k];
            }

            std::size_t start = 0;
            if (firstMap)
            {
                controls[0] = initialApplied;
                statuses[0] = initialStatus;
                start = 1;
                firstMap = false;
            }
            impl_->batch.run(count - start,
                             [&](std::size_t index, CBFControlFilter &filter)
                             {
                                 const std::size_t k = index + start;
                                 statuses[k] = filter.filter(guess[k], nominals[k], stepSize,
                                                             controls[k]);
                             });
            ++impl_->statistics.mapBatches;
            impl_->statistics.mapFilterCalls += count - start;
            filterCalls += count - start;
            for (std::size_t k = start; k < count; ++k)
            {
                filteredCalls += statuses[k] == CBFControlFilter::Status::Filtered ? 1u : 0u;
                blockedCalls += statuses[k] == CBFControlFilter::Status::Blocked ? 1u : 0u;
            }

            candidate[0] = out.end;
            for (unsigned int k = 0; k < count; ++k)
                candidate[k + 1] = Operations::integrate(candidate[k], controls[k], spans[k]);

            residual = 0.0;
            for (unsigned int k = 1; k <= count; ++k)
                residual = std::max(
                    residual, (candidate[k] - guess[k]).cwiseAbs().maxCoeff());
            impl_->statistics.maxResidual = std::max(impl_->statistics.maxResidual, residual);
            if (residual <= impl_->parameters.convergenceTolerance)
            {
                converged = true;
                break;
            }

            for (unsigned int k = 1; k <= count; ++k)
                guess[k] = (1.0 - impl_->parameters.relaxation) * guess[k] +
                           impl_->parameters.relaxation * candidate[k];
        }

        ++impl_->statistics.windows;
        impl_->statistics.convergedWindows += converged ? 1u : 0u;

        if (elapsed + windowBudget + 1e-12 >= budget &&
            Operations::difference(candidate.back(), to).cwiseAbs().maxCoeff() <= 1e-12)
            candidate.back() = Operations::normalize(to);

        // At an exact fixed point, each map evaluation was already made at the actual
        // emitted segment start and returned that segment's control. Reuse that proof
        // instead of paying for an identical verification batch. Bitwise equality is
        // deliberate: a tolerance-based match would weaken the safety contract.
        bool mapCertified = true;
        for (unsigned int k = 0; k <= count; ++k)
            mapCertified = mapCertified &&
                           FilteredStateSpace::bitwiseEqual(candidate[k], guess[k]);
        for (unsigned int k = 0; k < count; ++k)
            mapCertified = mapCertified && statuses[k] != CBFControlFilter::Status::Blocked;

        bool verifiedWindow = mapCertified;
        if (mapCertified)
        {
            ++impl_->statistics.mapCertifiedWindows;
        }
        else
        {
            // Otherwise the ordinary pointwise filter is evaluated at every actual
            // segment start. These calls are independent and retain a parallel path.
            std::vector<Configuration> verified(count);
            std::vector<CBFControlFilter::Status> verificationStatus(
                count, CBFControlFilter::Status::Blocked);
            impl_->batch.run(count,
                             [&](std::size_t k, CBFControlFilter &filter)
                             {
                                 const Control proposed =
                                     Operations::difference(candidate[k], candidate[k + 1]) /
                                     spans[k];
                                 verificationStatus[k] = filter.filter(
                                     candidate[k], proposed, stepSize, verified[k]);
                             });
            ++impl_->statistics.verificationBatches;
            impl_->statistics.verificationFilterCalls += count;
            filterCalls += count;

            verifiedWindow = true;
            for (unsigned int k = 0; k < count; ++k)
            {
                const Control proposed =
                    Operations::difference(candidate[k], candidate[k + 1]) / spans[k];
                filteredCalls +=
                    verificationStatus[k] == CBFControlFilter::Status::Filtered ? 1u : 0u;
                blockedCalls +=
                    verificationStatus[k] == CBFControlFilter::Status::Blocked ? 1u : 0u;
                if (verificationStatus[k] == CBFControlFilter::Status::Blocked ||
                    (verified[k] - proposed).cwiseAbs().maxCoeff() >
                        impl_->parameters.verificationTolerance ||
                    !finite(candidate[k + 1]))
                    verifiedWindow = false;
            }
        }
        if (!verifiedWindow)
        {
            ++impl_->statistics.verificationFailures;
            return fail();
        }

        for (unsigned int k = 1; k <= count; ++k)
        {
            out.travel += Operations::distance(candidate[k - 1], candidate[k], maxSpeed);
            if (!FilteredStateSpace::bitwiseEqual(out.waypoints.back(), candidate[k]))
                out.waypoints.push_back(candidate[k]);
        }
        out.end = candidate.back();
        elapsed += windowBudget;
    }

    if (out.waypoints.size() < 2)
        return fail();

    out.steps = static_cast<unsigned int>(filterCalls - blockedCalls);
    out.filtered = static_cast<unsigned int>(filteredCalls);
    out.blocked = static_cast<unsigned int>(blockedCalls);
    out.fraction = budget / horizon;
    out.reachedTarget = Operations::distance(out.end, to, maxSpeed) <= reachTolerance;
    ++impl_->statistics.accepted;
    impl_->statistics.seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return true;
}
