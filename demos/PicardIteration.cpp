#include <ompl/picard/PicardIteration.h>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

int main()
{
    using Solver = ompl::picard::PicardIteration<6, 32>;
    using State = Solver::State;

    ompl::picard::Options<double> options;
    options.nodeCount = 16;
    options.maxIterations = 8;
    options.horizon = 0.5;
    options.relaxation = 1.0;
    options.absoluteTolerance = 3e-6;
    options.relativeTolerance = 1e-7;

    Solver solver(options);
    const State start{0.0, -1.0, 0.5, -0.5, 0.25, 0.0};
    const State goal{1.0, 0.5, -0.5, 0.75, -0.25, 0.5};
    constexpr double gain = 1.5;

    auto vectorField = [&](const State &q, double, State &dqdt)
    {
        // Replace this body with the CBF-filtered control callback.
        for (std::size_t d = 0; d < q.size(); ++d)
            dqdt[d] = gain * (goal[d] - q[d]);
    };

    const auto result = solver.solveLinear(start, goal, vectorField);

    std::cout << "status=" << (result.converged() ? "converged" : "not-converged")
              << " iterations=" << result.iterations << " max_update=" << result.maxUpdate << '\n';
    std::cout << "endpoint:";
    for (const double value : solver.endpoint())
        std::cout << ' ' << std::fixed << std::setprecision(6) << value;
    std::cout << '\n';

    constexpr std::size_t runs = 100000;
    double checksum = 0.0;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t run = 0; run < runs; ++run)
    {
        solver.solveLinear(start, goal, vectorField);
        checksum += solver.endpoint()[run % start.size()];
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsedNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    std::cout << "average solve: " << elapsedNs / static_cast<double>(runs) << " ns"
              << " checksum=" << checksum << '\n';
    return 0;
}
