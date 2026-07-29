#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace ompl::picard
{
    enum class IntegrationRule
    {
        LeftRectangle,
        Trapezoidal
    };

    enum class Status
    {
        Converged,
        MaxIterations,
        InvalidOptions,
        NonFiniteVectorField,
        NonFiniteTrajectory
    };

    template <typename Scalar>
    struct Options
    {
        std::size_t nodeCount{16};
        std::size_t maxIterations{3};
        Scalar horizon{Scalar{1}};
        Scalar relaxation{Scalar{0.7}};
        Scalar absoluteTolerance{Scalar{1e-6}};
        Scalar relativeTolerance{Scalar{1e-6}};
        IntegrationRule integrationRule{IntegrationRule::Trapezoidal};
    };

    template <typename Scalar>
    struct Result
    {
        Status status{Status::InvalidOptions};
        std::size_t iterations{0};
        Scalar maxUpdate{std::numeric_limits<Scalar>::infinity()};

        [[nodiscard]] auto converged() const noexcept -> bool
        {
            return status == Status::Converged;
        }
    };

    /// Allocation-free scalar Picard iteration over a fixed-capacity trajectory.
    ///
    /// VectorField must be callable as:
    ///
    ///     field(const State &q, Scalar time, State &dqdt)
    ///
    /// All field evaluations in one Picard round read only the previous iterate,
    /// so this class is ready for later node-level parallelization without
    /// changing its mathematical update.
    template <std::size_t Dimension, std::size_t MaxNodes = 64, typename Scalar = double>
    class PicardIteration
    {
    public:
        static_assert(Dimension > 0, "Picard state dimension must be positive");
        static_assert(MaxNodes >= 2, "Picard trajectory needs at least two nodes");

        using State = std::array<Scalar, Dimension>;
        using Trajectory = std::array<State, MaxNodes>;
        using SolverOptions = Options<Scalar>;
        using SolverResult = Result<Scalar>;

        explicit PicardIteration(const SolverOptions &options = SolverOptions{}) noexcept
        {
            setOptions(options);
        }

        auto setOptions(const SolverOptions &options) noexcept -> bool
        {
            options_ = options;
            optionsValid_ = validateOptions(options_);
            if (optionsValid_)
                timeStep_ = options_.horizon / static_cast<Scalar>(options_.nodeCount - 1);
            else
                timeStep_ = Scalar{0};
            return optionsValid_;
        }

        [[nodiscard]] auto options() const noexcept -> const SolverOptions &
        {
            return options_;
        }

        [[nodiscard]] auto nodeCount() const noexcept -> std::size_t
        {
            return options_.nodeCount;
        }

        [[nodiscard]] auto timeStep() const noexcept -> Scalar
        {
            return timeStep_;
        }

        [[nodiscard]] auto timeAt(std::size_t index) const noexcept -> Scalar
        {
            return static_cast<Scalar>(index) * timeStep_;
        }

        /// Initialize the Picard iterate with a straight line from start to goal.
        void initializeLinear(const State &start, const State &goal) noexcept
        {
            activeBuffer_ = 0;
            initialized_ = optionsValid_;
            if (!initialized_)
                return;

            auto &trajectory = trajectories_[activeBuffer_];
            const Scalar denominator = static_cast<Scalar>(options_.nodeCount - 1);
            for (std::size_t i = 0; i < options_.nodeCount; ++i)
            {
                const Scalar alpha = static_cast<Scalar>(i) / denominator;
                for (std::size_t d = 0; d < Dimension; ++d)
                    trajectory[i][d] = start[d] + alpha * (goal[d] - start[d]);
            }
        }

        /// Initialize every trajectory node at the initial state.
        void initializeConstant(const State &initial) noexcept
        {
            activeBuffer_ = 0;
            initialized_ = optionsValid_;
            if (!initialized_)
                return;

            auto &trajectory = trajectories_[activeBuffer_];
            for (std::size_t i = 0; i < options_.nodeCount; ++i)
                trajectory[i] = initial;
        }

        /// Run Picard iteration from the current trajectory guess.
        ///
        /// The initial state is held exactly at node zero. The vector field is
        /// evaluated at every node of the previous iterate before any node of the
        /// new iterate is written (Jacobi/Picard semantics).
        template <typename VectorField>
        auto solveFromCurrentGuess(const State &initial, VectorField &&field) noexcept -> SolverResult
        {
            if (!optionsValid_ || !initialized_)
                return {Status::InvalidOptions, 0, std::numeric_limits<Scalar>::infinity()};

            for (std::size_t iteration = 0; iteration < options_.maxIterations; ++iteration)
            {
                auto &current = trajectories_[activeBuffer_];
                auto &next = trajectories_[1U - activeBuffer_];

                // Expensive phase: every call is independent within this round.
                for (std::size_t i = 0; i < options_.nodeCount; ++i)
                {
                    field(current[i], timeAt(i), vectorField_[i]);
                    if (!allFinite(vectorField_[i]))
                        return {Status::NonFiniteVectorField, iteration + 1, std::numeric_limits<Scalar>::infinity()};
                }

                next[0] = initial;
                State integral{};
                Scalar maxUpdate = maxAbsDifference(next[0], current[0]);
                Scalar maxMagnitude = maxAbs(next[0]);

                for (std::size_t i = 1; i < options_.nodeCount; ++i)
                {
                    accumulateIntegral(integral, vectorField_[i - 1], vectorField_[i]);

                    State candidate{};
                    for (std::size_t d = 0; d < Dimension; ++d)
                    {
                        candidate[d] = initial[d] + integral[d];
                        next[i][d] = current[i][d] + options_.relaxation * (candidate[d] - current[i][d]);
                    }

                    if (!allFinite(next[i]))
                        return {Status::NonFiniteTrajectory, iteration + 1, std::numeric_limits<Scalar>::infinity()};

                    maxUpdate = std::max(maxUpdate, maxAbsDifference(next[i], current[i]));
                    maxMagnitude = std::max(maxMagnitude, maxAbs(next[i]));
                }

                activeBuffer_ = 1U - activeBuffer_;

                const Scalar tolerance = options_.absoluteTolerance + options_.relativeTolerance * maxMagnitude;
                if (maxUpdate <= tolerance)
                    return {Status::Converged, iteration + 1, maxUpdate};
            }

            return {Status::MaxIterations, options_.maxIterations, lastUpdateNorm()};
        }

        template <typename VectorField>
        auto solveLinear(const State &initial, const State &goal, VectorField &&field) noexcept -> SolverResult
        {
            initializeLinear(initial, goal);
            return solveFromCurrentGuess(initial, std::forward<VectorField>(field));
        }

        template <typename VectorField>
        auto solveConstant(const State &initial, VectorField &&field) noexcept -> SolverResult
        {
            initializeConstant(initial);
            return solveFromCurrentGuess(initial, std::forward<VectorField>(field));
        }

        [[nodiscard]] auto trajectory() const noexcept -> const Trajectory &
        {
            return trajectories_[activeBuffer_];
        }

        [[nodiscard]] auto state(std::size_t index) const noexcept -> const State &
        {
            return trajectories_[activeBuffer_][index];
        }

        [[nodiscard]] auto endpoint() const noexcept -> const State &
        {
            return trajectories_[activeBuffer_][options_.nodeCount - 1];
        }

        [[nodiscard]] auto endpointError(const State &goal) const noexcept -> Scalar
        {
            return maxAbsDifference(endpoint(), goal);
        }

    private:
        static auto validateOptions(const SolverOptions &options) noexcept -> bool
        {
            return options.nodeCount >= 2 && options.nodeCount <= MaxNodes && options.maxIterations > 0 &&
                   std::isfinite(options.horizon) && options.horizon > Scalar{0} &&
                   std::isfinite(options.relaxation) && options.relaxation > Scalar{0} &&
                   options.relaxation <= Scalar{1} && std::isfinite(options.absoluteTolerance) &&
                   options.absoluteTolerance >= Scalar{0} && std::isfinite(options.relativeTolerance) &&
                   options.relativeTolerance >= Scalar{0};
        }

        static auto allFinite(const State &state) noexcept -> bool
        {
            for (const Scalar value : state)
                if (!std::isfinite(value))
                    return false;
            return true;
        }

        static auto maxAbs(const State &state) noexcept -> Scalar
        {
            Scalar result{0};
            for (const Scalar value : state)
                result = std::max(result, std::abs(value));
            return result;
        }

        static auto maxAbsDifference(const State &a, const State &b) noexcept -> Scalar
        {
            Scalar result{0};
            for (std::size_t d = 0; d < Dimension; ++d)
                result = std::max(result, std::abs(a[d] - b[d]));
            return result;
        }

        void accumulateIntegral(State &integral, const State &left, const State &right) const noexcept
        {
            if (options_.integrationRule == IntegrationRule::Trapezoidal)
            {
                const Scalar scale = Scalar{0.5} * timeStep_;
                for (std::size_t d = 0; d < Dimension; ++d)
                    integral[d] += scale * (left[d] + right[d]);
            }
            else
            {
                for (std::size_t d = 0; d < Dimension; ++d)
                    integral[d] += timeStep_ * left[d];
            }
        }

        [[nodiscard]] auto lastUpdateNorm() const noexcept -> Scalar
        {
            const auto &current = trajectories_[activeBuffer_];
            const auto &previous = trajectories_[1U - activeBuffer_];
            Scalar result{0};
            for (std::size_t i = 0; i < options_.nodeCount; ++i)
                result = std::max(result, maxAbsDifference(current[i], previous[i]));
            return result;
        }

        SolverOptions options_{};
        Scalar timeStep_{0};
        bool optionsValid_{false};
        bool initialized_{false};
        std::size_t activeBuffer_{0};
        std::array<Trajectory, 2> trajectories_{};
        Trajectory vectorField_{};
    };
}  // namespace ompl::picard
