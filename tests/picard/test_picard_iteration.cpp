#define BOOST_TEST_MODULE PicardIterationTest
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <limits>

#include <ompl/picard/PicardIteration.h>

namespace picard = ompl::picard;

BOOST_AUTO_TEST_CASE(ConstantVectorFieldIsIntegratedExactly)
{
    using Solver = picard::PicardIteration<2, 16>;
    using State = Solver::State;

    picard::Options<double> options;
    options.nodeCount = 9;
    options.maxIterations = 4;
    options.horizon = 2.0;
    options.relaxation = 1.0;
    options.absoluteTolerance = 1e-13;
    options.relativeTolerance = 0.0;

    Solver solver(options);
    const State initial{1.0, -1.0};
    const State velocity{0.5, -2.0};

    const auto result = solver.solveConstant(initial, [&](const State &, double, State &dqdt)
    {
        dqdt = velocity;
    });

    BOOST_CHECK(result.converged());
    BOOST_CHECK_EQUAL(result.iterations, 2);

    for (std::size_t i = 0; i < solver.nodeCount(); ++i)
    {
        const double t = solver.timeAt(i);
        BOOST_CHECK_SMALL(solver.state(i)[0] - (initial[0] + t * velocity[0]), 1e-12);
        BOOST_CHECK_SMALL(solver.state(i)[1] - (initial[1] + t * velocity[1]), 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(ContractiveLinearSystemConverges)
{
    using Solver = picard::PicardIteration<1, 64>;
    using State = Solver::State;

    picard::Options<double> options;
    options.nodeCount = 33;
    options.maxIterations = 12;
    options.horizon = 0.25;
    options.relaxation = 1.0;
    options.absoluteTolerance = 1e-12;
    options.relativeTolerance = 1e-12;

    Solver solver(options);
    const State initial{1.0};

    const auto result = solver.solveConstant(initial, [](const State &q, double, State &dqdt)
    {
        dqdt[0] = -q[0];
    });

    BOOST_CHECK(result.converged());
    BOOST_CHECK_SMALL(solver.endpoint()[0] - std::exp(-options.horizon), 2e-5);
}

BOOST_AUTO_TEST_CASE(LinearInitialGuessIsAvailableForSteering)
{
    using Solver = picard::PicardIteration<3, 16>;
    using State = Solver::State;

    picard::Options<double> options;
    options.nodeCount = 5;
    options.maxIterations = 1;
    options.horizon = 1.0;

    Solver solver(options);
    const State start{0.0, 1.0, 2.0};
    const State goal{4.0, 3.0, 0.0};
    solver.initializeLinear(start, goal);

    BOOST_CHECK_SMALL(solver.state(2)[0] - 2.0, 1e-12);
    BOOST_CHECK_SMALL(solver.state(2)[1] - 2.0, 1e-12);
    BOOST_CHECK_SMALL(solver.state(2)[2] - 1.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(NonFiniteVectorFieldIsRejected)
{
    using Solver = picard::PicardIteration<1, 8>;
    using State = Solver::State;

    picard::Options<double> options;
    options.nodeCount = 4;
    Solver solver(options);
    const State initial{0.0};

    const auto result = solver.solveConstant(initial, [](const State &, double t, State &dqdt)
    {
        dqdt[0] = t > 0.0 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    });

    BOOST_CHECK(result.status == picard::Status::NonFiniteVectorField);
}
