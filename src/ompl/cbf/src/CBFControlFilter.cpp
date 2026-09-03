#include "ompl/cbf/CBFControlFilter.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <qpmad/solver.h>

#include <ompl/cbf/Profiler.h>

namespace
{
    constexpr int nJoints = ompl::cbf::ClearanceBarrier::nJoints;
    // World spheres plus self-collision pairs. This is a *capacity*: screening decides
    // how many rows a given solve actually carries, and in open space it is a handful.
    constexpr int nConstraints = ompl::cbf::ClearanceBarrier::nConstraints;

    // Below this, a returned control counts as "the nominal, untouched". qpmad
    // returns the nominal bit-for-bit when nothing activates, so this only needs
    // to absorb rounding.
    constexpr double unchangedTolerance = 1e-12;
}  // namespace

/// All sizes are fixed at compile time, so a solve does no heap allocation.
struct ompl::cbf::CBFControlFilter::Solver
{
    using Backend = qpmad::SolverTemplate<double, nJoints, 1, nConstraints>;

    Backend backend;
    // qpmad factorizes the Hessian in place, so it gets a scratch copy each solve.
    Eigen::Matrix<double, nJoints, nJoints> hessian{Eigen::Matrix<double, nJoints, nJoints>::Zero()};
    Eigen::Matrix<double, nJoints, 1> objective;
    Eigen::Matrix<double, nConstraints, 1> rowLower;
    // One-sided rows: no upper limit on how fast clearance may grow.
    Eigen::Matrix<double, nConstraints, 1> rowUpper{
        Eigen::Matrix<double, nConstraints, 1>::Constant(std::numeric_limits<double>::infinity())};
    ClearanceBarrier::Evaluation evaluation;
    /// How fast each sphere's barrier can fall per unit time; see
    /// ClearanceBarrier::decreaseRates(). Constant, so computed once.
    ClearanceBarrier::Values decreaseRates;
    ClearanceBarrier::Values threshold;
    /// The horizon `threshold` was last scaled by. A rollout asks about the same step
    /// over and over, so rebuilding 343 scaled rows every call is pure repetition; the
    /// profile put it at 10% of a call. NaN forces the first build.
    double thresholdHorizon{std::numeric_limits<double>::quiet_NaN()};
};

ompl::cbf::CBFControlFilter::CBFControlFilter(const ClearanceBarrier &barrier)
  : CBFControlFilter(barrier, Parameters())
{
}

ompl::cbf::CBFControlFilter::CBFControlFilter(const ClearanceBarrier &barrier, const Parameters &parameters)
  : barrier_(barrier), parameters_(parameters), solver_(std::make_unique<Solver>())
{
    // Per unit time; scaled by the actual step in filter(), which is where dt is known.
    solver_->decreaseRates = barrier_.decreaseRates(parameters_.maxSpeed);
}

ompl::cbf::CBFControlFilter::~CBFControlFilter() = default;

void ompl::cbf::CBFControlFilter::controlBounds(const Configuration &q, double duration, Control &lower,
                                                Control &upper) const
{
    const Configuration jointLower = robots::UR5::lowerBounds();
    const Configuration jointUpper = robots::UR5::upperBounds();
    const double dt = duration;

    for (Eigen::Index j = 0; j < nJoints; ++j)
    {
        lower[j] = -parameters_.maxSpeed[j];
        upper[j] = parameters_.maxSpeed[j];

        if (parameters_.respectJointLimits)
        {
            // u_j must keep q_j + u_j*dt inside [jointLower, jointUpper].
            lower[j] = std::max(lower[j], (jointLower[j] - q[j]) / dt);
            upper[j] = std::min(upper[j], (jointUpper[j] - q[j]) / dt);
        }

        // A joint already outside its limits gives an empty interval. Freezing it
        // keeps the QP well posed and lets the remaining joints still move.
        if (lower[j] > upper[j])
            lower[j] = upper[j] = 0.0;
    }
}

ompl::cbf::ControlFilter::Status ompl::cbf::CBFControlFilter::filter(const Configuration &q,
                                                                    const Control &nominal, double duration,
                                                                    Control &filtered) const
{
    Diagnostics ignored;
    return filter(q, nominal, duration, filtered, ignored);
}

ompl::cbf::ControlFilter::Status ompl::cbf::CBFControlFilter::filter(const Configuration &q,
                                                                    const Control &nominal, double duration,
                                                                    Control &filtered,
                                                                    double &certified) const
{
    Diagnostics diagnostics;
    const Status status = filter(q, nominal, duration, filtered, diagnostics);
    certified = diagnostics.certifiedDuration;
    return status;
}

ompl::cbf::ControlFilter::Status ompl::cbf::CBFControlFilter::filter(const Configuration &q,
                                                                    const Control &nominal, double duration,
                                                                    Control &filtered, double &certified,
                                                                    double &safe) const
{
    Diagnostics diagnostics;
    const Status status = filter(q, nominal, duration, filtered, diagnostics);
    certified = diagnostics.certifiedDuration;
    // Both spans came out of the one pass `filter()` already made over the evaluation.
    // A blocked call leaves the control zero and certifies nothing either way.
    safe = status == Status::Blocked ? 0.0 : diagnostics.safeDuration;
    return status;
}

ompl::cbf::ControlFilter::Status ompl::cbf::CBFControlFilter::filter(const Configuration &q,
                                                                    const Control &nominal, double duration,
                                                                    Control &filtered,
                                                                    Diagnostics &diagnostics) const
{
    // A non-positive step has no meaningful CBF condition, and backward
    // propagation through a projection is not defined.
    if (duration <= 0.0)
    {
        filtered.setZero();
        FilterStats::instance().record(FilterOutcome::Blocked, 0, 0, 0.0);
        return Status::Blocked;
    }

    Solver &solver = *solver_;
    ClearanceBarrier::Evaluation &evaluation = solver.evaluation;
    if (parameters_.screening)
    {
        // Two horizons, and the row has to clear both. A sphere with h > rate/kappa is
        // satisfied by every control in the box, so its row cannot change the QP; a
        // sphere with h > rate*dt cannot reach zero across the step the caller is about
        // to integrate. Screening on the larger keeps the QP's answer identical to the
        // unscreened one and the skipped spheres safe -- and costs no gradient and no
        // Jacobian, which is where the money is.
        const double horizon = parameters_.kappa > 0.0 ? 1.0 / parameters_.kappa
                                                       : std::numeric_limits<double>::infinity();
        const double scale = std::max(duration, horizon);
        if (!(scale == solver.thresholdHorizon))
        {
            solver.threshold = solver.decreaseRates * scale;
            solver.thresholdHorizon = scale;
        }
        ScopedTimer timer("evaluate_screened");
        barrier_.evaluateScreened(q, solver.threshold, evaluation);
    }
    else
    {
        ScopedTimer timer("evaluate_full");
        barrier_.evaluate(q, evaluation);
    }

    // Free: the region reads the values and boundaries this evaluation already
    // produced, and no gradient. Computed before the early returns below so a Blocked
    // call still reports where it may safely move, which is the one thing a caller
    // that has just been refused actually wants to know.
    diagnostics.region = barrier_.certifiedRegion(evaluation);
    diagnostics.worstValue = evaluation.values[static_cast<Eigen::Index>(evaluation.worst)];
    diagnostics.worstSphere = evaluation.worst;
    diagnostics.worstSelfValue =
        evaluation.values[ClearanceBarrier::nSpheres + static_cast<Eigen::Index>(evaluation.worstPair)];
    diagnostics.worstSelfPair = evaluation.worstPair;
    diagnostics.inBounds = evaluation.inBounds;
    diagnostics.solverIterations = 0;
    diagnostics.activeRows = evaluation.active;
    // Nothing is certified until a control has been settled on; every path that gives
    // up below leaves both at zero, which asks the caller to come back rather than run.
    // The gain is the same statement upside down, so its "certifies nothing" is
    // infinity: a blocked call must not be read as one that needed no allowance.
    diagnostics.certifiedDuration = 0.0;
    diagnostics.safeDuration = 0.0;
    diagnostics.requiredGain = std::numeric_limits<double>::infinity();

    // Outside the baked field, GridSDF clamps and over-reports clearance, so the
    // barrier cannot be trusted. Refusing to move is the only safe answer.
    if (!evaluation.inBounds)
    {
        filtered.setZero();
        FilterStats::instance().record(FilterOutcome::Blocked, diagnostics.activeRows, 0, 0.0);
        return Status::Blocked;
    }

    Control lower;
    Control upper;
    controlBounds(q, duration, lower, upper);

    // Continuous-time CBF: (dh_i/dq) u >= -kappa h_i, with no step length in it. Row r
    // constrains barrier evaluation.constraint[r], which is r itself unless screening
    // reordered things -- and which may be a world sphere or a self-collision pair,
    // indifferently.
    const Eigen::Index active = evaluation.active;
    for (Eigen::Index r = 0; r < active; ++r)
        solver.rowLower[r] = evaluation.values[evaluation.constraint[r]] * -parameters_.kappa;

    if (active == 0)
    {
        // Screening has proved that no barrier can bind during this step. What remains
        // is a diagonal box QP, whose exact solution is the component-wise clamp of the
        // nominal control. Avoid entering qpmad for this common open-space case.
        filtered = nominal.cwiseMax(lower).cwiseMin(upper);
    }
    else
    {
        // minimize 0.5 u^T W u - (W uNom)^T u, i.e. H = W and objective = -W uNom.
        solver.hessian.diagonal() = parameters_.weights;
        solver.objective = -parameters_.weights.cwiseProduct(nominal);

        try
        {
            // Fixed capacity, variable occupancy: qpmad's template arguments are maxima and
            // it reads the constraint count off the matrix it is handed, so passing fewer
            // rows costs less without allocating anything.
            ScopedTimer qpTimer("qp_solve");
            const auto status = solver.backend.solve(filtered, solver.hessian, solver.objective, lower, upper,
                                                     evaluation.rows.topRows(active),
                                                     solver.rowLower.head(active),
                                                     solver.rowUpper.head(active));
            diagnostics.solverIterations = solver.backend.getNumberOfInequalityIterations();
            if (status != Solver::Backend::OK)
            {
                filtered.setZero();
                FilterStats::instance().record(FilterOutcome::Blocked, diagnostics.activeRows,
                                               diagnostics.solverIterations, 0.0);
                return Status::Blocked;
            }
        }
        catch (const std::exception &)
        {
            // qpmad signals infeasibility by throwing: the robot is cornered, and no
            // control satisfies every clearance row.
            filtered.setZero();
            FilterStats::instance().record(FilterOutcome::Blocked, diagnostics.activeRows,
                                           diagnostics.solverIterations, 0.0);
            return Status::Blocked;
        }
    }

    // How long the control just chosen could be run for. The clearances are already in
    // hand, so this is a matvec; see ClearanceBarrier::certifiedDuration() for why it is
    // a statement about the whole interval and not an extrapolation.
    //
    // The joint limits need re-checking against it. The QP's control box keeps
    // q + u*duration inside them, which says nothing about a longer span, and running
    // out of joint travel is not something the barrier can see coming.
    {
        ScopedTimer certTimer("certified_duration");
        barrier_.durations(evaluation, filtered, parameters_.kappa, diagnostics.safeDuration,
                           diagnostics.certifiedDuration);
    }

    // The lexicographic program's second stage, and it is this cheap because the first
    // stage was the QP above: the least gain certifying `filtered` is the reciprocal of
    // how long `filtered` may safely run, which `durations()` has just computed. Taken
    // before the joint-limit clamp below deliberately -- running out of joint travel
    // shortens a step without any barrier having asked for a faster decay, and charging
    // the gain for it would report an allowance the rows never wanted.
    //
    // Identical to `ClearanceBarrier::requiredGain(diagnostics.region, filtered)`, which
    // is where the derivation lives; spelled as the reciprocal here to avoid repeating
    // that function's matvec on a number already in hand.
    diagnostics.requiredGain = diagnostics.safeDuration > 0.0
                                   ? 1.0 / diagnostics.safeDuration
                                   : std::numeric_limits<double>::infinity();

    if (parameters_.respectJointLimits)
    {
        const Configuration jointLower = robots::UR5::lowerBounds();
        const Configuration jointUpper = robots::UR5::upperBounds();
        for (Eigen::Index j = 0; j < nJoints; ++j)
        {
            if (filtered[j] == 0.0)
                continue;
            const double room = (filtered[j] > 0.0 ? jointUpper[j] : jointLower[j]) - q[j];
            const double travel = std::max(room / filtered[j], 0.0);
            diagnostics.certifiedDuration = std::min(diagnostics.certifiedDuration, travel);
            diagnostics.safeDuration = std::min(diagnostics.safeDuration, travel);
        }
    }

    const Status status = (filtered - nominal).norm() <= unchangedTolerance ? Status::Unchanged
                                                                             : Status::Filtered;
    FilterStats::instance().record(status == Status::Unchanged ? FilterOutcome::Unchanged
                                                                : FilterOutcome::Filtered,
                                   diagnostics.activeRows, diagnostics.solverIterations,
                                   diagnostics.certifiedDuration, diagnostics.requiredGain);
    return status;
}
