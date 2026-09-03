#pragma once

#include <cstddef>
#include <limits>
#include <memory>

#include <Eigen/Core>

#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/cbf/ControlFilter.h>

namespace ompl::cbf
{
    /// A control barrier function safety filter, solved as a small QP.
    ///
    /// The nominal control is projected onto the set of controls that keep every
    /// collision sphere's clearance from decaying too fast:
    ///
    ///     minimize    0.5 (u - uNom)^T W (u - uNom)
    ///     subject to  (dh_i/dq) u  >=  -kappa * h_i(q)           for every sphere i
    ///                 uMin <= u <= uMax
    ///
    /// The constraint is the continuous-time CBF condition `dh_i/dt >= -kappa h_i`,
    /// whose solutions satisfy `h_i(t) >= h_i(0) e^{-kappa t}`: the safe set is
    /// forward invariant, and the rate at which clearance may be spent is a property
    /// of the filter rather than of whatever step a caller happens to ask about.
    /// `kappa` has units of 1/s. Larger is more permissive — as `kappa -> inf` the
    /// row disappears and only the control box remains — and `kappa = 0` forbids any
    /// decay at all. The linearization and geometric errors are absorbed by
    /// `ClearanceBarrier`'s margin, not here.
    ///
    /// ### Why not the discrete-time form
    ///
    /// Earlier revisions enforced `h_i(q + u dt) >= (1 - gamma) h_i(q)`, linearized,
    /// which is this row with `kappa = gamma / dt`. That coupling is the problem: the
    /// permitted decay *rate* then floats with the step a caller asks about, so halving
    /// `dt` doubles how fast clearance may be spent per second and the condition does
    /// not converge to anything as `dt -> 0`. It also makes the certificate and the
    /// screening threshold answer questions in different units from the row they came
    /// from. `decayRate()` converts an old per-step `gamma` at a reference step into the
    /// rate that reproduces it exactly at that step.
    ///
    /// What the change does *not* do is make the implementation continuous-time: the
    /// row is still imposed at sampled configurations with a control held constant in
    /// between (`ParallelPicardRollout`), so the inter-sample guarantee still comes from
    /// `ClearanceBarrier`'s margin and from `certifiedDuration()`, which is now an
    /// interval statement about the same exponential envelope the row promises.
    ///
    /// ### Why the QP is cheap
    ///
    /// Six variables, forty one-sided rows, and a diagonal positive-definite
    /// Hessian. qpmad's dual active set starts from the *unconstrained* minimum,
    /// which for this objective is exactly `uNom`, and activates constraints only
    /// as they turn out to be violated. So a step that is already safe returns
    /// `uNom` untouched at almost no cost, and there is no need to guess in advance
    /// which spheres matter: the solver finds the active ones itself. That removes
    /// both the heuristic active-set preselection and the constraint-refinement
    /// loop that a primal solver needs.
    ///
    /// ### Screening, and what it trades
    ///
    /// Building the forty rows costs about three times the solve, so the rows are
    /// where the money is. `screening` (on by default) uses
    /// `ClearanceBarrier::decreaseRates()` to drop, before computing anything, the
    /// spheres whose clearance provably cannot reach zero within the step — leaving
    /// them at the cost of one interpolated distance each, which is what a plain
    /// collision check would have paid anyway.
    ///
    /// Under the continuous-time row this is exact rather than a trade. A row whose
    /// clearance exceeds `rate_i / kappa` is satisfied by every control in the box, so
    /// it cannot change the QP's feasible set and dropping it changes nothing; the
    /// threshold is `rate_i * max(dt, 1/kappa)`, the larger horizon also covering the
    /// separate requirement that a dropped row stay non-negative across the step the
    /// caller integrates. The discrete-time version screened at `rate_i * dt` alone,
    /// which was short of what its own row needed and so gave up the decay guarantee on
    /// the spheres it skipped. Turning screening off now changes cost, not semantics.
    ///
    /// ### The certificate, which is the same bound spent differently
    ///
    /// Screening asks "which rows can bind within this step?" and usually answers
    /// "none". The five-argument `filter()` asks the complementary question — "how
    /// long until one could?" — and hands the answer back as
    /// `Diagnostics::certifiedDuration`. Over that span the filter is provably a
    /// no-op, so a caller integrating the returned control across it gets the motion
    /// repeated filtering would have produced, without the calls. In open space that
    /// collapses a whole tree extension into one evaluation and one straight line;
    /// in clutter the certificate is short and the caller steps as it always did.
    ///
    /// It costs one 40x6 matvec on top of an evaluation that has already happened, so
    /// it is always computed. What it is *not* is a licence to take a longer QP step:
    /// the linearization error the margin absorbs still grows with the step, which is
    /// why the certificate is a Lipschitz bound over the interval rather than an
    /// extrapolation of the rows.
    ///
    /// ### The adaptive gain, which costs a division
    ///
    /// The gain can be treated as a decision variable rather than a constant: among the
    /// controls that deviate least from the nominal one, prefer the one needing the least
    /// allowance to spend clearance. Written out, that is the lexicographic program
    ///
    ///     lexmin_{u, kappa}  ( 0.5 (u - uNom)^T W (u - uNom),  kappa )
    ///     subject to         (dh_i/dq) u >= -kappa h_i,  u in U,  0 <= kappa <= kappaMax
    ///
    /// and it decomposes, which is the whole point. Because every `h_i > 0` on the safe
    /// set, row i reads `kappa >= -(dh_i/dq) u / h_i`, so a control is feasible for *some*
    /// gain at or below the cap exactly when it is feasible *at* the cap. The first
    /// stage's feasible set is therefore the fixed-gain one at `kappa = kappaMax`, and its
    /// minimiser is the fixed-gain QP's — `W` positive definite makes it unique, so the
    /// second stage has a single control to price and no search to do.
    ///
    /// Two things follow, and both are worth being blunt about.
    ///
    /// The first is that `kappa` here *is* `kappaMax`: this class already solves stage
    /// one, and no parameter was added because none is needed. The second is that
    /// **the returned control is the same vector either way**. Minimising the gain does
    /// not bend the trajectory; it reports what the trajectory cost. So an A/B between
    /// "adaptive" and "fixed at the cap" would measure nothing, and the value is not
    /// speed or reach but the number itself: `Diagnostics::requiredGain`, the gain the
    /// step actually needed, which tightens the decay envelope an audit can claim and
    /// gives `kappa` something to be tuned against other than taste.
    ///
    /// It is read off the certified region rather than the gradients — see
    /// `ClearanceBarrier::requiredGain()` for why a region can answer a question about
    /// rows, and for the upper-bound direction that makes doing so sound. Having the
    /// region answer it is what makes the gain free: it is the reciprocal of a span this
    /// class computes anyway, so the second stage is a division, and it is computed on
    /// every call for the same reason the certificate is.
    ///
    /// ### Threading
    ///
    /// Not thread safe: the solver and its scratch space are reused across calls.
    /// Give each thread its own filter.
    class CBFControlFilter : public ControlFilter
    {
    public:
        struct Parameters
        {
            /// CBF decay rate in 1/s: `dh/dt >= -kappa h`, so `h(t) >= h(0) e^{-kappa t}`.
            /// Smaller is more conservative; 0 forbids any decay. Independent of the step
            /// a caller asks about — see `decayRate()` to convert an old per-step gamma.
            /// The default is what `gamma = 0.99` amounted to at a 0.05 s step.
            ///
            /// This is also the `kappaMax` of the adaptive-gain program, which needs no
            /// parameter of its own: see the class comment for why a cap and a fixed gain
            /// are the same number here, and `Diagnostics::requiredGain` for what the step
            /// spent of it.
            double kappa{19.8};
            /// Diagonal of W: the relative cost of deviating on each joint.
            Control weights{Control::Ones()};
            /// Per-joint speed limit, |u_j| <= maxSpeed_j.
            Control maxSpeed{robots::UR5::velocityLimits()};
            /// Also constrain u so that q + u*dt stays inside the joint limits.
            bool respectJointLimits{true};
            /// Skip constraint rows for spheres that cannot bind within the step.
            /// See the class comment for what this gives up.
            bool screening{true};
        };

        /// Optional per-call detail, for diagnostics and benchmarking.
        struct Diagnostics
        {
            double worstValue{0.0};            ///< min_i h_i(q) over the world spheres
            std::size_t worstSphere{0};        ///< which sphere that was
            double worstSelfValue{0.0};        ///< min_p h_ab(q) over the self-collision pairs
            std::size_t worstSelfPair{0};      ///< which pair that was
            bool inBounds{true};               ///< were all centers inside the SDF?
            std::ptrdiff_t solverIterations{0};  ///< qpmad active-set iterations
            int activeRows{ClearanceBarrier::nConstraints};  ///< rows that survived screening
            /// How long the returned control stays certified; see `ControlFilter`'s
            /// five-argument `filter()` and `ClearanceBarrier::certifiedDuration()`.
            double certifiedDuration{0.0};
            /// The longer, weaker span: how long \p filtered may run before any barrier
            /// could reach zero, against `certifiedDuration`'s "before any row could
            /// bind". Both come from one pass; see `ClearanceBarrier::durations()`.
            double safeDuration{0.0};
            /// The smallest gain that would have produced this control: the adaptive
            /// gain, read off the certified region by `ClearanceBarrier::requiredGain()`.
            ///
            /// `kappa` is the *cap*; this is what the step actually needed, and it is
            /// usually far below. Aggregated over a rollout it says what gain a plan
            /// would have run at, which is the number to tune `kappa` from and the one an
            /// audit should quote: over the returned control every barrier obeys
            /// `h_i(t) >= h_i e^{-requiredGain t}`, a tighter envelope than `kappa`
            /// promises.
            ///
            /// Infinite on a `Blocked` call, and on any call whose region certifies
            /// nothing -- consistent with the two durations, which report zero for the
            /// same reason. It may exceed `kappa` even on a call the QP accepted, since
            /// the region bounds the rows more coarsely than the gradients the QP solved
            /// against; that is the region being honest about what *it* can prove, not a
            /// violation of the row the filter enforced.
            double requiredGain{std::numeric_limits<double>::infinity()};
            /// The certified region at the configuration this call evaluated, which
            /// falls out of that evaluation for free -- it reads the values and
            /// boundaries already in hand and needs no gradient. A caller wanting the
            /// longer, weaker span asks it for `ClearanceBarrier::safeDuration()`; see
            /// there for what separates the two.
            ClearanceBarrier::CertifiedRegion region;
        };

        /// The decay rate that reproduces the old per-step condition
        /// `h(q + u dt) >= (1 - gamma) h(q)` exactly at a step of \p dt, for porting a
        /// tuned `gamma` across. The row was `(dh/dq) u >= -gamma h / dt`, so the rate is
        /// `gamma / dt` and nothing is approximated at that step; what changes is that
        /// the rate now stays put when the step does not.
        static double decayRate(double gamma, double dt)
        {
            return gamma / dt;
        }

        /// \p barrier is not copied and must outlive this filter.
        explicit CBFControlFilter(const ClearanceBarrier &barrier);
        CBFControlFilter(const ClearanceBarrier &barrier, const Parameters &parameters);
        ~CBFControlFilter() override;

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered) const override;

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered, double &certified) const override;

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &filtered, double &certified, double &safe) const override;

        /// As above, additionally reporting why.
        Status filter(const Configuration &q, const Control &nominal, double duration, Control &filtered,
                      Diagnostics &diagnostics) const;

        const char *name() const override
        {
            return "cbf-qp";
        }

        const Parameters &parameters() const
        {
            return parameters_;
        }

        void setParameters(const Parameters &parameters)
        {
            parameters_ = parameters;
        }

        const ClearanceBarrier &barrier() const
        {
            return barrier_;
        }

        /// The control box actually enforced at \p q for a step of \p duration: the
        /// speed limit intersected with what the joint limits allow over that step.
        /// Exposed because a directed control sampler wants to clamp its nominal
        /// control the same way.
        void controlBounds(const Configuration &q, double duration, Control &lower, Control &upper) const;

    private:
        struct Solver;  // hides qpmad from this header

        const ClearanceBarrier &barrier_;
        Parameters parameters_;
        std::unique_ptr<Solver> solver_;
    };
}  // namespace ompl::cbf
