#define BOOST_TEST_MODULE CBFControlFilterTest
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/cbf/CBFControlFilter.h>
#include <ompl/util/RandomNumbers.h>

using Barrier = ompl::cbf::ClearanceBarrier;
using Filter = ompl::cbf::CBFControlFilter;
using ompl::cbf::ControlFilter;
using UR5 = ompl::robots::UR5;
namespace sdf = ompl::sdf;

namespace
{
    const Eigen::Vector3d obstacleCenter(0.3, 0.3, 1.1);
    constexpr double obstacleRadius = 0.2;
    constexpr double voxel = 0.02;

    sdf::DistanceFn sphereField(const Eigen::Vector3d &center, double radius)
    {
        return [center, radius](const Eigen::Vector3d &p) { return (p - center).norm() - radius; };
    }

    Eigen::AlignedBox3d reachableBox()
    {
        return Eigen::AlignedBox3d(Eigen::Vector3d(-1.1, -1.1, 0.35), Eigen::Vector3d(1.1, 1.1, 2.05));
    }

    /// An obstacle close enough to bind at the zero configuration.
    const sdf::GridSDF &nearField()
    {
        static const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), reachableBox(), voxel);
        return field;
    }

    /// An obstacle far enough away that no clearance row ever binds, so tests can
    /// isolate the control box.
    const sdf::GridSDF &farField()
    {
        static const sdf::GridSDF field(sphereField(Eigen::Vector3d(8.0, 8.0, 8.0), 0.2), reachableBox(),
                                        voxel);
        return field;
    }

    /// The step length every test propagates over.
    constexpr double duration = 0.05;

    /// Rates are quoted per second; 8 /s is what the old per-step gamma of 0.4 came to
    /// at `duration`, so the numbers these tests assert against are unchanged.
    Filter::Parameters parameters(double kappa = 8.0)
    {
        Filter::Parameters p;
        p.kappa = kappa;
        // Generous, so the clearance rows rather than the box are what bind.
        p.maxSpeed = UR5::Configuration::Constant(10.0);
        p.respectJointLimits = false;
        return p;
    }

    /// A nominal control aimed straight at the tightest obstacle, scaled by \p speed.
    UR5::Configuration towardWorstSphere(const Barrier &barrier, const UR5::Configuration &q, double speed)
    {
        const Barrier::Evaluation evaluation = barrier.evaluate(q);
        const Eigen::Index worst = static_cast<Eigen::Index>(evaluation.worst);
        return -evaluation.rows.row(worst).transpose().normalized() * speed;
    }
}  // namespace

// qpmad's dual active set starts from the unconstrained minimum, which for this
// objective is the nominal control. A safe step therefore costs zero active-set
// iterations and comes back bit-for-bit.
BOOST_AUTO_TEST_CASE(SafeNominalPassesThroughUntouched)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter filter(barrier, parameters());

    const UR5::Configuration q = UR5::Configuration::Zero();
    const UR5::Configuration nominal = towardWorstSphere(barrier, q, 1.0);

    UR5::Configuration filtered;
    Filter::Diagnostics diagnostics;
    BOOST_CHECK(filter.filter(q, nominal, duration, filtered, diagnostics) == ControlFilter::Status::Unchanged);
    BOOST_CHECK_EQUAL((filtered - nominal).norm(), 0.0);
    BOOST_CHECK_EQUAL(diagnostics.solverIterations, 0);
    BOOST_CHECK(diagnostics.inBounds);
}

// With one row active the QP has a closed form: projecting uNom onto
// {u : a^T u >= b} under the identity metric gives u = a*b/|a|^2. Because the
// nominal here points straight down -a, that result is *independent of how
// aggressive the nominal is* -- the filter always lands on the same boundary point.
BOOST_AUTO_TEST_CASE(ProjectionMatchesTheClosedForm)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter::Parameters p = parameters();
    const Filter filter(barrier, p);

    const UR5::Configuration q = UR5::Configuration::Zero();
    const Barrier::Evaluation evaluation = barrier.evaluate(q);
    const Eigen::Index worst = static_cast<Eigen::Index>(evaluation.worst);
    const UR5::Configuration a = evaluation.rows.row(worst).transpose();
    const double b = -p.kappa * evaluation.values[worst];
    const UR5::Configuration expected = a * b / a.squaredNorm();

    for (double speed : {2.0, 4.0, 10.0, 20.0})
    {
        UR5::Configuration filtered;
        BOOST_CHECK(filter.filter(q, towardWorstSphere(barrier, q, speed), duration, filtered) ==
                    ControlFilter::Status::Filtered);
        BOOST_CHECK_LE((filtered - expected).norm(), 1e-9);
        // The row it activated ends up exactly tight.
        BOOST_CHECK_LE(std::abs(a.dot(filtered) - b), 1e-9);
    }
}

// Every row must hold, not just the one that bound.
BOOST_AUTO_TEST_CASE(EveryClearanceRowIsSatisfied)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter::Parameters p = parameters();
    const Filter filter(barrier, p);

    const UR5::Configuration q = UR5::Configuration::Zero();
    const Barrier::Evaluation evaluation = barrier.evaluate(q);

    UR5::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, towardWorstSphere(barrier, q, 10.0), duration, filtered) ==
                  ControlFilter::Status::Filtered);

    // rows * u >= -kappa * h, for all spheres.
    const Barrier::Values slack = evaluation.rows * filtered + evaluation.values * p.kappa;
    BOOST_CHECK_GE(slack.minCoeff(), -1e-9);
}

// The point of the whole exercise: a nominal control that would drive the arm into the
// obstacle is replaced by one that spends clearance no faster than `kappa`, and the true
// clearance -- not just the linearized prediction -- holds up.
BOOST_AUTO_TEST_CASE(ClearanceDecayIsCappedByKappa)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter::Parameters p = parameters();
    const Filter filter(barrier, p);

    const UR5::Configuration q = UR5::Configuration::Zero();
    const double before = barrier.worstValue(q);
    BOOST_REQUIRE_GT(before, 0.0);

    const UR5::Configuration nominal = towardWorstSphere(barrier, q, 10.0);
    UR5::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, nominal, duration, filtered) == ControlFilter::Status::Filtered);

    // Unfiltered, this step ends in collision.
    BOOST_CHECK_LT(barrier.worstValue(q + nominal * duration), 0.0);

    // Filtered, clearance stays above the CBF floor. The row bounds dh/dt, so over a
    // step held constant it delivers the first-order term of the exponential envelope,
    // `(1 - kappa dt) h`, which is the weaker of the two and so the right thing to assert.
    // The 1 mm slack covers step linearization; measured error here is about 0.3 mm.
    const double after = barrier.worstValue(q + filtered * duration);
    BOOST_CHECK_GE(after, (1.0 - p.kappa * duration) * before - 1e-3);
    BOOST_CHECK_GT(after, 0.0);
}

BOOST_AUTO_TEST_CASE(SmallerKappaIsMoreConservative)
{
    const UR5 robot;
    // Self rows are switched off for this one, and they have to be. The claim under test is
    // about the row the step is being driven *into* -- `nominal` aims at the worst sphere
    // against `nearField()` -- and kappa decides how fast that row's clearance may be
    // spent. Leave the self rows in and a different row binds at some rates but not
    // others, so `worstValue()` stops tracking the row the experiment is about and the
    // sequence is not monotone in kappa. That is not kappa failing to be conservative; it
    // is the measurement changing what it measures. A large negative margin puts every self
    // row far out of reach so none can ever be the minimum.
    const Barrier barrier(robot, nearField(), /*margin=*/0.0, /*selfMargin=*/-100.0);
    const UR5::Configuration q = UR5::Configuration::Zero();
    const UR5::Configuration nominal = towardWorstSphere(barrier, q, 10.0);

    double previous = -1.0;
    for (double kappa : {4.0, 8.0, 16.0, 20.0})
    {
        const Filter::Parameters p = parameters(kappa);
        const Filter filter(barrier, p);
        UR5::Configuration filtered;
        BOOST_REQUIRE(filter.filter(q, nominal, duration, filtered) == ControlFilter::Status::Filtered);

        // A larger rate permits more decay, so the resulting clearance shrinks.
        const double after = barrier.worstValue(q + filtered * duration);
        if (previous >= 0.0)
            BOOST_CHECK_LT(after, previous);
        previous = after;
    }
}

BOOST_AUTO_TEST_CASE(SpeedBoxIsRespected)
{
    const UR5 robot;
    const Barrier barrier(robot, farField(), /*margin=*/0.0);
    Filter::Parameters p = parameters();
    p.maxSpeed = UR5::velocityLimits();  // 0.5 rad/s
    const Filter filter(barrier, p);

    const UR5::Configuration q = UR5::Configuration::Zero();
    UR5::Configuration filtered;
    BOOST_CHECK(filter.filter(q, UR5::Configuration::Constant(50.0), duration, filtered) ==
                ControlFilter::Status::Filtered);
    for (Eigen::Index j = 0; j < filtered.size(); ++j)
        BOOST_CHECK_LE(std::abs(filtered[j]), p.maxSpeed[j] + 1e-12);
}

// With screening and no surviving barrier rows, the QP reduces exactly to a
// component-wise clamp. Exercise both the untouched and clamped cases so the
// solver-free fast path remains equivalent to the diagonal box QP.
BOOST_AUTO_TEST_CASE(ZeroActiveRowsUseTheExactBoxProjection)
{
    const UR5 robot;
    // Move self-collision rows far outside the screening band as well as the world
    // obstacle, making zero active rows deterministic rather than scene-dependent.
    const Barrier barrier(robot, farField(), /*margin=*/0.0, /*selfMargin=*/-100.0);
    Filter::Parameters p = parameters();
    p.maxSpeed = UR5::velocityLimits();
    p.respectJointLimits = false;
    p.screening = true;
    const Filter filter(barrier, p);

    const UR5::Configuration q = UR5::Configuration::Zero();
    Filter::Diagnostics diagnostics;
    UR5::Configuration filtered;

    const UR5::Configuration inside = UR5::Configuration::Constant(0.25);
    BOOST_CHECK(filter.filter(q, inside, duration, filtered, diagnostics) ==
                ControlFilter::Status::Unchanged);
    BOOST_CHECK_EQUAL(diagnostics.activeRows, 0);
    BOOST_CHECK_EQUAL((filtered - inside).norm(), 0.0);

    UR5::Configuration outside;
    outside << -2.0, -0.25, 0.75, 0.0, 3.0, -3.0;
    const UR5::Configuration expected = outside.cwiseMax(-p.maxSpeed).cwiseMin(p.maxSpeed);
    BOOST_CHECK(filter.filter(q, outside, duration, filtered, diagnostics) ==
                ControlFilter::Status::Filtered);
    BOOST_CHECK_EQUAL(diagnostics.activeRows, 0);
    BOOST_CHECK_EQUAL((filtered - expected).norm(), 0.0);
}

BOOST_AUTO_TEST_CASE(JointLimitsAreRespected)
{
    const UR5 robot;
    const Barrier barrier(robot, farField(), /*margin=*/0.0);
    Filter::Parameters p = parameters();
    p.respectJointLimits = true;
    const Filter filter(barrier, p);

    // Start a hair below the upper limit and push hard against it.
    const UR5::Configuration upper = UR5::upperBounds();
    UR5::Configuration q = UR5::Configuration::Zero();
    q[0] = upper[0] - 0.001;

    UR5::Configuration filtered;
    BOOST_REQUIRE(filter.filter(q, UR5::Configuration::Constant(10.0), duration, filtered) !=
                  ControlFilter::Status::Blocked);

    const UR5::Configuration next = q + filtered * duration;
    for (Eigen::Index j = 0; j < next.size(); ++j)
        BOOST_CHECK_LE(next[j], upper[j] + 1e-12);
    BOOST_CHECK_LE(filtered[0], 0.001 / duration + 1e-12);
}

// Outside the baked field GridSDF clamps and over-reports clearance, so there is
// no trustworthy barrier. That must read as "cannot move", never as "safe".
BOOST_AUTO_TEST_CASE(LeavingTheFieldBlocksTheStep)
{
    const UR5 robot;
    const Eigen::AlignedBox3d tiny(Eigen::Vector3d(-0.2, -0.2, 0.8), Eigen::Vector3d(0.2, 0.2, 1.0));
    const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), tiny, voxel);
    const Barrier barrier(robot, field);
    const Filter filter(barrier, parameters());

    UR5::Configuration filtered;
    Filter::Diagnostics diagnostics;
    BOOST_CHECK(filter.filter(UR5::Configuration::Zero(), UR5::Configuration::Constant(0.1), duration,
                              filtered, diagnostics) == ControlFilter::Status::Blocked);
    BOOST_CHECK_EQUAL(filtered.norm(), 0.0);
    BOOST_CHECK(!diagnostics.inBounds);
}

// The base sphere's row is identically zero -- no joint can move it -- so an
// obstacle swallowing the base makes the QP infeasible. qpmad signals that by
// throwing, and the filter has to turn it into a clean "blocked".
BOOST_AUTO_TEST_CASE(CorneredReportsBlockedRatherThanThrowing)
{
    const UR5 robot;
    const UR5::Configuration q = UR5::Configuration::Zero();
    const Eigen::Vector3d base = robot.sphereCenters(q).col(0);
    const sdf::GridSDF field(sphereField(base, 0.5), reachableBox(), voxel);
    const Barrier barrier(robot, field, /*margin=*/0.0);
    const Filter filter(barrier, parameters());

    UR5::Configuration filtered;
    BOOST_CHECK_NO_THROW(filter.filter(q, UR5::Configuration::Constant(0.1), duration, filtered));
    BOOST_CHECK(filter.filter(q, UR5::Configuration::Constant(0.1), duration, filtered) ==
                ControlFilter::Status::Blocked);
    BOOST_CHECK_EQUAL(filtered.norm(), 0.0);
}

// PathControl::check() re-propagates an edge and compares against the stored
// state, so a non-deterministic filter would make every CBF path fail validation.
BOOST_AUTO_TEST_CASE(FilterIsDeterministic)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter filter(barrier, parameters());

    const UR5::Configuration q = UR5::Configuration::Zero();
    const UR5::Configuration nominal = towardWorstSphere(barrier, q, 10.0);

    UR5::Configuration first;
    UR5::Configuration second;
    filter.filter(q, nominal, duration, first);
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        filter.filter(q, nominal, duration, second);
        BOOST_CHECK_EQUAL((first - second).norm(), 0.0);
    }
}

BOOST_AUTO_TEST_CASE(PassthroughFilterIsIdentity)
{
    const ompl::cbf::PassthroughFilter filter;
    const UR5::Configuration nominal = UR5::Configuration::Constant(123.0);
    UR5::Configuration filtered;
    BOOST_CHECK(filter.filter(UR5::Configuration::Zero(), nominal, duration, filtered) ==
                ControlFilter::Status::Unchanged);
    BOOST_CHECK_EQUAL((filtered - nominal).norm(), 0.0);
}

BOOST_AUTO_TEST_CASE(ControlBoundsIntersectSpeedAndJointLimits)
{
    const UR5 robot;
    const Barrier barrier(robot, farField());
    Filter::Parameters p = parameters();
    p.maxSpeed = UR5::Configuration::Constant(0.5);
    p.respectJointLimits = true;
    const Filter filter(barrier, p);

    UR5::Configuration q = UR5::Configuration::Zero();
    q[1] = UR5::lowerBounds()[1] + 0.002;  // near the lower limit

    UR5::Configuration lower;
    UR5::Configuration upper;
    filter.controlBounds(q, duration, lower, upper);

    BOOST_CHECK_LE(lower[1], upper[1]);
    BOOST_CHECK_CLOSE(lower[1], -0.002 / duration, 1e-9);  // joint limit binds
    BOOST_CHECK_CLOSE(upper[1], 0.5, 1e-9);                  // speed limit binds
    BOOST_CHECK_CLOSE(lower[0], -0.5, 1e-9);                 // mid-range joint: speed only
    BOOST_CHECK_CLOSE(upper[0], 0.5, 1e-9);
}

// Screening skips constraint rows that cannot bind. Two things have to hold for that to
// be worth having: the control must be the one the full 40-row solve would have produced,
// and the row count must actually drop.
//
// Under the continuous-time row the first is exact rather than statistical. A row is
// dropped only when its clearance exceeds `rate/kappa`, at which point every control in
// the box satisfies it, so it cannot change the feasible set and cannot change the argmin
// of a strictly convex objective. The discrete-time predecessor screened at `rate*dt`,
// which was short of that, and could only claim agreement most of the time.
BOOST_AUTO_TEST_CASE(ScreeningMatchesTheFullSolveAndDropsMostRows)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), 0.0);

    // Realistic speeds: the screen's threshold scales with maxSpeed, and at the
    // deliberately generous 10 rad/s the other tests use, nothing is ever screened.
    Filter::Parameters base = parameters();
    base.maxSpeed = UR5::velocityLimits();
    Filter::Parameters screenedParameters = base;
    screenedParameters.screening = true;
    Filter::Parameters fullParameters = base;
    fullParameters.screening = false;

    const Filter screened(barrier, screenedParameters);
    const Filter full(barrier, fullParameters);

    ompl::RNG rng;
    long rows = 0;
    int steps = 0, agreed = 0;
    for (int sample = 0; sample < 1500; ++sample)
    {
        UR5::Configuration q, nominal;
        for (Eigen::Index j = 0; j < 6; ++j)
        {
            q[j] = rng.uniformReal(UR5::lowerBounds()[j], UR5::upperBounds()[j]);
            nominal[j] = rng.uniformReal(-base.maxSpeed[j], base.maxSpeed[j]);
        }

        UR5::Configuration screenedControl, fullControl;
        Filter::Diagnostics diagnostics;
        const ControlFilter::Status status =
            screened.filter(q, nominal, duration, screenedControl, diagnostics);
        if (status == ControlFilter::Status::Blocked)
            continue;
        full.filter(q, nominal, duration, fullControl);

        // Admissible whatever else is true.
        for (Eigen::Index j = 0; j < 6; ++j)
            BOOST_REQUIRE_LE(std::abs(screenedControl[j]), base.maxSpeed[j] + 1e-9);

        rows += diagnostics.activeRows;
        ++steps;
        if ((screenedControl - fullControl).norm() <= 1e-9)
            ++agreed;
    }

    BOOST_REQUIRE_GT(steps, 200);
    const double meanRows = static_cast<double>(rows) / steps;
    const double agreement = static_cast<double>(agreed) / steps;
    BOOST_TEST_MESSAGE("screened rows " << meanRows << " of 40, agreement " << agreement);
    BOOST_CHECK_LT(meanRows, 25.0);
    BOOST_CHECK_EQUAL(agreed, steps);
}

// The claim `requiredGain` makes, checked against the gradients it never looked at:
// every row -- including the ones screening dropped -- holds at the gain reported, so a
// caller may quote the tighter envelope `h_i(t) >= h_i e^{-requiredGain t}` in place of
// the one `kappa` promises. Randomised, because the interesting failures are poses where
// the lever-arm bound is loose in an unexpected direction.
BOOST_AUTO_TEST_CASE(RequiredGainSatisfiesEveryRowItReports)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    Filter::Parameters p = parameters();
    p.maxSpeed = UR5::velocityLimits();
    const Filter filter(barrier, p);

    ompl::RNG rng;
    int steps = 0, belowCap = 0;
    double worstSlack = std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < 1500; ++sample)
    {
        UR5::Configuration q, nominal;
        for (Eigen::Index j = 0; j < 6; ++j)
        {
            q[j] = rng.uniformReal(UR5::lowerBounds()[j], UR5::upperBounds()[j]);
            nominal[j] = rng.uniformReal(-p.maxSpeed[j], p.maxSpeed[j]);
        }

        UR5::Configuration filtered;
        Filter::Diagnostics diagnostics;
        if (filter.filter(q, nominal, duration, filtered, diagnostics) == ControlFilter::Status::Blocked)
            continue;

        // The gain is a statement about the safe set, and at `margin = 0` over the whole
        // joint range most random poses are not in it. A row that is already negative has
        // no slack for the region to divide into, so the honest report is infinity -- no
        // gain certifies a barrier that has already gone. Those poses are not what this
        // test is about; the ones inside the set are.
        const Barrier::Evaluation full = barrier.evaluate(q);
        if (full.values.minCoeff() <= 0.0 || diagnostics.region.slack.minCoeff() <= 0.0)
            continue;
        ++steps;

        BOOST_REQUIRE(std::isfinite(diagnostics.requiredGain));
        BOOST_REQUIRE_GE(diagnostics.requiredGain, 0.0);

        // The rows, from a full evaluation -- gradients for all 40, screened or not.
        for (Eigen::Index i = 0; i < Barrier::nConstraints; ++i)
        {
            const double slack =
                full.rows.row(i).dot(filtered) + diagnostics.requiredGain * full.values[i];
            worstSlack = std::min(worstSlack, slack);
            BOOST_REQUIRE_GE(slack, -1e-9);
        }

        if (diagnostics.requiredGain < p.kappa)
            ++belowCap;
    }

    BOOST_REQUIRE_GT(steps, 200);
    BOOST_TEST_MESSAGE("worst row slack at requiredGain " << worstSlack << ", below cap "
                                                          << belowCap << "/" << steps);
    // The point of reporting it: the gain a step needs is usually well under the cap.
    BOOST_CHECK_GT(static_cast<double>(belowCap) / steps, 0.5);
}

// It is the reciprocal of the safety span, which is what makes the second stage of the
// lexicographic program a division rather than a solve. Both spellings, since the filter
// takes the cheap one and callers get the explicit one.
BOOST_AUTO_TEST_CASE(RequiredGainIsTheReciprocalOfTheSafeSpan)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    const Filter filter(barrier, parameters());

    const UR5::Configuration q = UR5::Configuration::Zero();
    for (double speed : {0.5, 1.0, 4.0, 10.0})
    {
        UR5::Configuration filtered;
        Filter::Diagnostics diagnostics;
        BOOST_REQUIRE(filter.filter(q, towardWorstSphere(barrier, q, speed), duration, filtered,
                                    diagnostics) != ControlFilter::Status::Blocked);

        BOOST_CHECK_CLOSE(diagnostics.requiredGain, 1.0 / diagnostics.safeDuration, 1e-9);
        BOOST_CHECK_CLOSE(diagnostics.requiredGain,
                          Barrier::requiredGain(diagnostics.region, filtered), 1e-9);
    }
}

// Screening drops *rows*; the region reads values and boundaries, which are filled for
// every constraint either way. So the gain a screened solve reports is the gain a full
// solve reports, bit for bit -- which is the reason to read it off the region at all.
BOOST_AUTO_TEST_CASE(RequiredGainIsUnaffectedByScreening)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    Filter::Parameters base = parameters();
    base.maxSpeed = UR5::velocityLimits();
    Filter::Parameters screenedParameters = base;
    screenedParameters.screening = true;
    Filter::Parameters fullParameters = base;
    fullParameters.screening = false;

    const Filter screened(barrier, screenedParameters);
    const Filter full(barrier, fullParameters);

    ompl::RNG rng;
    int steps = 0;
    for (int sample = 0; sample < 500; ++sample)
    {
        UR5::Configuration q, nominal;
        for (Eigen::Index j = 0; j < 6; ++j)
        {
            q[j] = rng.uniformReal(UR5::lowerBounds()[j], UR5::upperBounds()[j]);
            nominal[j] = rng.uniformReal(-base.maxSpeed[j], base.maxSpeed[j]);
        }

        UR5::Configuration screenedControl, fullControl;
        Filter::Diagnostics screenedDiagnostics, fullDiagnostics;
        if (screened.filter(q, nominal, duration, screenedControl, screenedDiagnostics) ==
            ControlFilter::Status::Blocked)
            continue;
        full.filter(q, nominal, duration, fullControl, fullDiagnostics);
        ++steps;

        BOOST_REQUIRE_EQUAL(screenedDiagnostics.requiredGain, fullDiagnostics.requiredGain);
    }
    BOOST_REQUIRE_GT(steps, 100);
}

// A control that moves no clearance needs no allowance to spend it; a call that was
// refused certifies nothing and must not be read as the former. Zero and infinity are
// the two ends of the same statement, and confusing them would let an audit quote a
// gain of zero for a step the filter would not take.
BOOST_AUTO_TEST_CASE(RequiredGainBracketsTheDegenerateCases)
{
    const UR5 robot;
    const Barrier barrier(robot, farField(), /*margin=*/0.0);
    const Filter filter(barrier, parameters());

    const UR5::Configuration q = UR5::Configuration::Zero();
    UR5::Configuration filtered;
    Filter::Diagnostics diagnostics;
    BOOST_REQUIRE(filter.filter(q, UR5::Configuration::Zero(), duration, filtered, diagnostics) ==
                  ControlFilter::Status::Unchanged);
    BOOST_CHECK_EQUAL(diagnostics.requiredGain, 0.0);

    // Out of the baked field: blocked before the QP, and nothing is certified.
    const Eigen::AlignedBox3d tiny(Eigen::Vector3d(-0.05, -0.05, 0.35), Eigen::Vector3d(0.05, 0.05, 0.45));
    const sdf::GridSDF field(sphereField(obstacleCenter, obstacleRadius), tiny, voxel);
    const Barrier clipped(robot, field, /*margin=*/0.0);
    const Filter blocking(clipped, parameters());
    Filter::Diagnostics blockedDiagnostics;
    BOOST_REQUIRE(blocking.filter(q, UR5::Configuration::Zero(), duration, filtered,
                                  blockedDiagnostics) == ControlFilter::Status::Blocked);
    BOOST_CHECK(std::isinf(blockedDiagnostics.requiredGain));
}

// How much the region overstates the gain, which the class comment leaves to
// measurement. The gradients give the exact answer -- `max_i [-(dh_i/dq) u]_+ / h_i` --
// and the region must never come in under it, or the envelope it licenses is a fiction.
// The ratio between them is the price of not computing a gradient, and it is reported
// rather than asserted tightly: it is a property of the lever-arm table, which is a
// worst case over the whole configuration space, so it is large by construction and
// changes whenever the sphere model does.
BOOST_AUTO_TEST_CASE(RequiredGainUpperBoundsTheGradientAnswer)
{
    const UR5 robot;
    const Barrier barrier(robot, nearField(), /*margin=*/0.0);
    Filter::Parameters p = parameters();
    p.maxSpeed = UR5::velocityLimits();
    const Filter filter(barrier, p);

    const double lipschitz = std::max(nearField().maxGradientNorm(), 1.0);
    ompl::RNG rng;
    int steps = 0, wanted = 0;
    int bindingClearance = 0, bindingBoundary = 0, bindingSelf = 0;
    double ratioSum = 0.0, ratioMax = 0.0;
    for (int sample = 0; sample < 1500; ++sample)
    {
        UR5::Configuration q, nominal;
        for (Eigen::Index j = 0; j < 6; ++j)
        {
            q[j] = rng.uniformReal(UR5::lowerBounds()[j], UR5::upperBounds()[j]);
            nominal[j] = rng.uniformReal(-p.maxSpeed[j], p.maxSpeed[j]);
        }

        UR5::Configuration filtered;
        Filter::Diagnostics diagnostics;
        if (filter.filter(q, nominal, duration, filtered, diagnostics) == ControlFilter::Status::Blocked)
            continue;
        const Barrier::Evaluation full = barrier.evaluate(q);
        if (full.values.minCoeff() <= 0.0 || diagnostics.region.slack.minCoeff() <= 0.0)
            continue;
        ++steps;

        double exact = 0.0;
        for (Eigen::Index i = 0; i < Barrier::nConstraints; ++i)
            exact = std::max(exact, -full.rows.row(i).dot(filtered) / full.values[i]);

        // Which term of `slack` set the answer. A world row's slack is
        // `min(h_i/L, boundary_i)`, and the two mean different things: clearance is the
        // obstacle, boundary is the edge of the baked field. If the gain were usually
        // set by the second, it would be reporting the size of the SDF box rather than
        // anything about the scene.
        const auto travel = (Barrier::leverArms() * filtered.cwiseAbs()).eval();
        Eigen::Index binding = 0;
        double bindingRatio = -1.0;
        for (Eigen::Index i = 0; i < Barrier::nConstraints; ++i)
            if (travel[i] > 0.0 && travel[i] / diagnostics.region.slack[i] > bindingRatio)
            {
                bindingRatio = travel[i] / diagnostics.region.slack[i];
                binding = i;
            }
        if (binding >= Barrier::nSpheres)
            ++bindingSelf;
        else if (full.boundary[binding] < full.values[binding] / lipschitz)
            ++bindingBoundary;
        else
            ++bindingClearance;

        // The bound direction is the assertion; everything else here is a measurement.
        BOOST_REQUIRE_GE(diagnostics.requiredGain, exact - 1e-9);

        if (exact > 0.0)
        {
            const double ratio = diagnostics.requiredGain / exact;
            ratioSum += ratio;
            ratioMax = std::max(ratioMax, ratio);
            ++wanted;
        }
    }

    BOOST_REQUIRE_GT(steps, 200);
    BOOST_TEST_MESSAGE("region/gradient gain ratio: mean " << (wanted ? ratioSum / wanted : 0.0)
                                                          << ", max " << ratioMax << " over " << wanted
                                                          << " of " << steps << " steps");
    BOOST_TEST_MESSAGE("binding row was clearance " << bindingClearance << ", sdf-box boundary "
                                                    << bindingBoundary << ", self-pair " << bindingSelf);
}
