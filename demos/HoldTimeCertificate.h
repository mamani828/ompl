#pragma once
// Hold-time self-pair certificate, extracted from holdbench.cpp (validated there
// against exact FK: worst certified - true contact = -2.81e-04 over 400 configs).
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/robots/UR5.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace holdtime
{
using Robot = ompl::robots::UR5;
using Barrier = ompl::cbf::ClearanceBarrier;
using Configuration = Robot::Configuration;
using Vec3 = Eigen::Vector3d;
static constexpr int NJ = 6;

// -------------------------------------------------------------- instrumentation

/// Wall time spent inside the hold query, process-wide, for answering "how much of
/// the planner's time is the certificate?". Off unless OMPL_HOLD_TIMING is set,
/// because a clock read is a few percent of a query this size.
struct HoldTiming
{
    double seconds{0.0};
    std::size_t calls{0};
    std::size_t selfRows{0}, worldRows{0};
};

inline HoldTiming &holdTiming()
{
    static HoldTiming t;
    return t;
}

inline bool holdTimingEnabled()
{
    static const bool on = []
    {
        const char *v = std::getenv("OMPL_HOLD_TIMING");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    return on;
}
#ifdef HOLDTIME_COUNTERS
struct Counters
{
    std::size_t queries{0};
    std::size_t fk{0}, centers{0}, sdfBatch{0};
    std::size_t selfDistances{0}, selfSqrt{0}, worldRows{0};
    std::size_t cacheBuilds{0}, framePrefix{0}, endpointEnvelopes{0};
    std::size_t selfExpensive{0}, worldExpensive{0};
    std::size_t normalRoots{0}, isoRoots{0}, rootIterations{0};
    std::size_t candidates{0}, zeroReturns{0};
    std::size_t seedWorld{0}, noSeed{0}, worldCandidates{0}, selfCandidates{0};
    std::size_t trigPairs{0}, anchorEnvelopes{0}, emptyActive{0};
};
inline Counters &counters()
{
    static Counters c;
    return c;
}
#define HT_COUNT(field, n) (holdtime::counters().field += (n))
#else
#define HT_COUNT(field, n) ((void)0)
#endif

/// Which term of a row's `max(L1, local-L1, speed-capped, anchored)` produced that
/// row's certificate. A world row's answer is the largest of four sound lower
/// bounds, so the question "was the expensive term worth computing" is answered by
/// which one attained the max -- and ties are resolved toward the *cheapest* term
/// that attains it, because if L1 alone already reached the answer then the anchored
/// root solve bought nothing for that row.
///
/// `Anchored` names different arithmetic in the two families: a self-pair row anchors
/// on the pair's own contact normal (`NormalAnchor`), while a world row has no usable
/// normal -- the SDF is not convex -- and falls back to the isotropic bound
/// (`IsoAnchor`). Attribution is therefore only meaningful split by family.
enum class HoldTerm : unsigned char
{
    None = 0,      ///< no row bound the answer, so no term did either
    L1,            ///< the global lever-arm time, which every row has for free
    LocalL1,       ///< `localWorldRate`: telescoping local lever arms, no envelope needed
    SpeedCapped,   ///< `speedCappedTime`: initial speed, bounded acceleration, speed cap
    Anchored,      ///< the root solve: `NormalAnchor` for self pairs, `IsoAnchor` for world
};

/// World rows whose certificate was settled by the local-lever term before any
/// envelope was built, against those that reached the expensive path at all. The
/// ratio is what says whether the term pays for itself.
inline std::size_t &localPrunedTally()
{
    static std::size_t n = 0;
    return n;
}
inline std::size_t &localReachedTally()
{
    static std::size_t n = 0;
    return n;
}

/// How many queries were answered by the empty-screened-set shortcut. Always
/// compiled: the envelope filter reports it whether or not the counters are on.
inline std::size_t &emptyActiveTally()
{
    static std::size_t n = 0;
    return n;
}
inline std::size_t emptyActiveCount()
{
    return emptyActiveTally();
}

inline void resetCounters()
{
#ifdef HOLDTIME_COUNTERS
    counters() = Counters();
#endif
}

inline void reportCounters(std::FILE *out)
{
#ifdef HOLDTIME_COUNTERS
    const Counters &c = counters();
    const double n = c.queries > 0 ? static_cast<double>(c.queries) : 1.0;
    std::fprintf(out, "\noptimized per-query counters (%zu engine calls)\n", c.queries);
    std::fprintf(out, "  forward kinematics        %8.3f\n", c.fk / n);
    std::fprintf(out, "  sphere-centre builds      %8.3f\n", c.centers / n);
    std::fprintf(out, "  SDF batch calls           %8.3f\n", c.sdfBatch / n);
    std::fprintf(out, "  self pair distances       %8.3f\n", c.selfDistances / n);
    std::fprintf(out, "  self pair square roots    %8.3f\n", c.selfSqrt / n);
    std::fprintf(out, "  world L1 rows processed   %8.3f\n", c.worldRows / n);
    std::fprintf(out, "  HoldCache builds          %8.3f\n", c.cacheBuilds / n);
    std::fprintf(out, "  frame-prefix builds       %8.3f\n", c.framePrefix / n);
    std::fprintf(out, "  endpoint envelopes        %8.3f\n", c.endpointEnvelopes / n);
    std::fprintf(out, "  ...of which anchored      %8.3f\n", c.anchorEnvelopes / n);
    std::fprintf(out, "  expensive self rows       %8.3f\n", c.selfExpensive / n);
    std::fprintf(out, "  expensive world rows      %8.3f\n", c.worldExpensive / n);
    std::fprintf(out, "  normal root solves        %8.3f\n", c.normalRoots / n);
    std::fprintf(out, "  isotropic root solves     %8.3f\n", c.isoRoots / n);
    std::fprintf(out, "  root iterations           %8.3f\n", c.rootIterations / n);
    std::fprintf(out, "  candidates after seed     %8.3f\n", c.candidates / n);
    std::fprintf(out, "  blocked (returned zero)   %8.3f\n", c.zeroReturns / n);
    std::fprintf(out, "  world-seeded / unseeded   %8.3f %.3f\n", c.seedWorld / n,
                 c.noSeed / n);
    std::fprintf(out, "  candidates self / world   %8.3f %.3f\n", c.selfCandidates / n,
                 c.worldCandidates / n);
    std::fprintf(out, "  envelope sin/cos pairs    %8.3f\n", c.trigPairs / n);
#else
    (void)out;
#endif
}


// ---------------------------------------------------------------- scalar roots

/// sin and cos of the same argument in one call where the toolchain offers it.
static inline void sinCos(double x, double &s, double &c)
{
#if defined(__GNUC__) && !defined(__clang__)
    __builtin_sincos(x, &s, &c);
#else
    s = std::sin(x);
    c = std::cos(x);
#endif
}

static inline double speedCappedTime(double d0, double S, double C, double V)
{
    if (d0 <= 0.0) return 0.0;
    if (V <= 0.0) return std::numeric_limits<double>::infinity();
    if (C == 0.0) return S > 0.0 ? d0 / S : std::numeric_limits<double>::infinity();
    const double tc = (V - S) / C;
    const double dc = S * tc + 0.5 * C * tc * tc;
    if (d0 <= dc)
        return (2.0 * d0) / (S + std::sqrt(S * S + 2.0 * C * d0));
    return tc + (d0 - dc) / V;
}

// max_{0<=s<=t} of b*s + c*s^2/2
static inline double prefixMaxQuad(double t, double b, double c)
{
    double best = std::max(0.0, b * t + 0.5 * c * t * t);
    if (c < 0.0)
    {
        const double s = -b / c;
        if (s > 0.0 && s < t) best = std::max(best, b * s + 0.5 * c * s * s);
    }
    return best;
}

// max_{0<=s<=t} ||s*u + s^2*a/2||
static inline double prefixMaxNorm(double t, const Vec3 &u, const Vec3 &a)
{
    const double aa = a.dot(a), ua = u.dot(a), uu = u.dot(u);
    double best = (t * u + 0.5 * t * t * a).norm();
    if (aa > 0.0)
    {
        const double disc = 9.0 * ua * ua - 8.0 * aa * uu;
        if (disc >= 0.0)
        {
            const double rt = std::sqrt(disc);
            for (double r : {(-3.0 * ua - rt) / (2.0 * aa), (-3.0 * ua + rt) / (2.0 * aa)})
                if (r > 0.0 && r < t) best = std::max(best, (r * u + 0.5 * r * r * a).norm());
        }
    }
    return best;
}

// 1.5 rad / 2^12 = 3.7e-4 rad, well under minAdvance = 0.01 rad.
static int kBisect = 12;

// first t in [0,T] where prefixMaxNorm(t,u,a) + H t^3/6 reaches d0
static inline double anchoredTime(double d0, const Vec3 &u, const Vec3 &a, double H, double T)
{
    HT_COUNT(isoRoots, 1);
    if (prefixMaxNorm(T, u, a) + H * T * T * T / 6.0 <= d0) return T;
    HT_COUNT(rootIterations, kBisect);
    double lo = 0.0, hi = T;
    for (int i = 0; i < kBisect; ++i)
    {
        const double m = 0.5 * (lo + hi);
        if (prefixMaxNorm(m, u, a) + H * m * m * m / 6.0 <= d0) lo = m; else hi = m;
    }
    return lo;
}

// same, projected onto the contact normal: b = n.u0, c = n.a0
static inline double normalAnchoredTime(double d0, double b, double c, double H, double T)
{
    HT_COUNT(normalRoots, 1);
    if (prefixMaxQuad(T, b, c) + H * T * T * T / 6.0 <= d0) return T;
    HT_COUNT(rootIterations, kBisect);
    double lo = 0.0, hi = T;
    for (int i = 0; i < kBisect; ++i)
    {
        const double m = 0.5 * (lo + hi);
        if (prefixMaxQuad(m, b, c) + H * m * m * m / 6.0 <= d0) lo = m; else hi = m;
    }
    return lo;
}

// ------------------------------------------------------- per-hop envelope cache

struct HoldCache
{
    double Q[NJ][NJ];
    double lo[NJ][NJ], hi[NJ][NJ];
    double Om[NJ][NJ], chi[NJ][NJ], A[NJ][NJ], W[NJ][NJ];
    double cumV[NJ][NJ + 1], cumC[NJ][NJ + 1], cumH[NJ][NJ + 1];
    Vec3 omega[NJ][NJ], alpha[NJ][NJ];
    Vec3 cumU[NJ][NJ + 1], cumA[NJ][NJ + 1];
    double segLen[NJ];
    Vec3 seg[NJ];
    bool frameReady_[NJ]{};
    const Robot::Kinematics *kin_{nullptr};
    Configuration v_{Configuration::Zero()};

    // One O(n^2) pass. Pairwise cosines are frame-invariant, so this serves
    // EVERY frame window -- all 303 pairs read out of it.
    //
    // \p bases is the largest chain-base frame any consumer will ask about. The
    // prefix blocks below are per base frame and independent of each other, and the
    // UR5's pair table only ever bases a window at frames 0 to 3, so building 4 of
    // them rather than 6 is a third of this loop that nothing was going to read.
    /// Everything, eagerly: what `pairHold()` and `holdWorldScale()` below expect.
    void build(const Robot::Kinematics &kin, const Configuration &v, double T, int bases = NJ - 1)
    {
        buildLazy(kin, v, T);
        for (int f = 0; f <= bases; ++f)
            ensureFrame(f);
    }

    /// The frame-independent half. The prefix block for a chain-base frame is then
    /// built on first use: the blocks are independent of each other and a query reads
    /// two or three of the four the UR5's pair table can ask for, so building them on
    /// demand keeps the rest off the hot path.
    void buildLazy(const Robot::Kinematics &kin, const Configuration &v, double T)
    {
        kin_ = &kin;
        v_ = v;
        for (int f = 0; f < NJ; ++f)
            frameReady_[f] = false;
        buildShared(kin, v, T);
    }

    void ensureFrame(int f)
    {
        if (!frameReady_[f])
            buildFrame(f);
    }

    void buildShared(const Robot::Kinematics &kin, const Configuration &v, double T)
    {
        double cos0[NJ][NJ], sin0[NJ][NJ];
        for (int k = 0; k < NJ; ++k)
        {
            Q[k][k] = v[k] * v[k];
            lo[k][k] = hi[k][k] = 1.0;
            for (int m = k + 1; m < NJ; ++m)
            {
                const double c = std::clamp(kin.jointAxis[k].dot(kin.jointAxis[m]), -1.0, 1.0);
                cos0[k][m] = c;
                sin0[k][m] = std::sqrt(std::max(0.0, 1.0 - c * c));
            }
        }
        for (int gap = 1; gap < NJ; ++gap)
            for (int k = 0; k + gap < NJ; ++k)
            {
                const int m = k + gap;
                const double interior = gap > 1 ? Q[k + 1][m - 1] : 0.0;
                const double w = T * std::sqrt(std::max(0.0, interior));
                const double c = cos0[k][m], s = sin0[k][m];
                double L, Hh;
                if (w <= 0.0) { L = Hh = c; }
                else if (w >= M_PI) { L = -1.0; Hh = 1.0; }
                else
                {
                    // cos/sin(theta0 +/- w) without acos: the sign of the sine
                    // says whether the angle ran past pi (or below 0), which is
                    // where the cosine endpoint has to clamp.
                    HT_COUNT(trigPairs, 1);
                    // One sine-cosine pair, not two library calls on the same
                    // argument: the envelope build is almost entirely this.
                    double sw, cw;
                    sinCos(w, sw, cw);
                    const double cp = c * cw - s * sw, sp = s * cw + c * sw;
                    const double cm = c * cw + s * sw, sm = s * cw - c * sw;
                    L = sp < 0.0 ? -1.0 : cp;
                    Hh = sm < 0.0 ? 1.0 : cm;
                }
                lo[k][m] = lo[m][k] = L;
                hi[k][m] = hi[m][k] = Hh;
                const double vp = v[k] * v[m];
                const double pr = vp * (vp >= 0.0 ? Hh : L);
                Q[k][m] = std::max(0.0, Q[k + 1][m] + Q[k][m - 1] - interior + 2.0 * pr);
            }

        for (int j = 0; j + 1 < NJ; ++j)
        {
            seg[j] = kin.jointOrigin[j + 1] - kin.jointOrigin[j];
            segLen[j] = seg[j].norm();
        }

    }

    // Prefix quantities depend only on the BASE frame f, so six of these
    // cover all 21 possible frame windows.
    void buildFrame(int f)
    {
        HT_COUNT(framePrefix, 1);
        frameReady_[f] = true;
        const Robot::Kinematics &kin = *kin_;
        const Configuration &v = v_;
        {
            for (int j = f; j < NJ; ++j) Om[f][j] = std::sqrt(Q[f][j]);
            chi[f][f] = 0.0;
            for (int k = f + 1; k < NJ; ++k)
            {
                double l = 0.0, h = 0.0;
                for (int r = f; r < k; ++r)
                {
                    if (v[r] >= 0.0) { l += v[r] * lo[r][k]; h += v[r] * hi[r][k]; }
                    else             { l += v[r] * hi[r][k]; h += v[r] * lo[r][k]; }
                }
                const double ma = std::max(std::max(l, -h), 0.0);
                chi[f][k] = std::sqrt(std::max(0.0, Q[f][k - 1] - ma * ma));
            }
            double acc = 0.0;
            for (int j = f; j < NJ; ++j) { acc += std::fabs(v[j]) * chi[f][j]; A[f][j] = acc; }
            W[f][f] = 0.0;
            double wacc = 0.0;
            for (int k = f + 1; k < NJ; ++k)
            {
                wacc += std::fabs(v[k]) * (A[f][k - 1] + Om[f][k - 1] * chi[f][k]);
                W[f][k] = wacc;
            }
            Vec3 om = Vec3::Zero(), al = Vec3::Zero();
            for (int j = f; j < NJ; ++j)
            {
                al += v[j] * om.cross(kin.jointAxis[j]);   // om is still omega_{j-1}
                om += v[j] * kin.jointAxis[j];
                omega[f][j] = om;
                alpha[f][j] = al;
            }
            cumV[f][f] = cumC[f][f] = cumH[f][f] = 0.0;
            cumU[f][f] = cumA[f][f] = Vec3::Zero();
            for (int j = f; j + 1 < NJ; ++j)
            {
                const double L = segLen[j], o = Om[f][j], a2 = A[f][j];
                cumV[f][j + 1] = cumV[f][j] + L * o;
                cumC[f][j + 1] = cumC[f][j] + L * (a2 + o * o);
                cumH[f][j + 1] = cumH[f][j] + L * (W[f][j] + 3.0 * a2 * o + o * o * o);
                cumU[f][j + 1] = cumU[f][j] + omega[f][j].cross(seg[j]);
                cumA[f][j + 1] = cumA[f][j] + alpha[f][j].cross(seg[j])
                                 + omega[f][j].cross(omega[f][j].cross(seg[j]));
            }
        }
    }

    // O(1) per pair: three multiply-adds off the prefix sums plus the last segment.
    double pairHold(const Robot::Kinematics &kin, const Robot::SphereCenters &centers,
                    std::size_t a, std::size_t b, int f, int g, double d0, double T,
                    bool withIsotropicAnchored) const
    {
        const int j = g - 1;
        const Vec3 pb = centers.col(static_cast<Eigen::Index>(b));
        const Vec3 last = pb - kin.jointOrigin[j];
        const double L = last.norm();
        const double o = Om[f][j], a2 = A[f][j];
        double V = cumV[f][j] + L * o;
        const double C = cumC[f][j] + L * (a2 + o * o);
        const double H = cumH[f][j] + L * (W[f][j] + 3.0 * a2 * o + o * o * o);
        const Vec3 u0 = cumU[f][j] + omega[f][j].cross(last);
        const Vec3 a0 = cumA[f][j] + alpha[f][j].cross(last)
                        + omega[f][j].cross(omega[f][j].cross(last));
        const double S = u0.norm();
        V = std::max(V, S);

        double best = std::min(T, speedCappedTime(d0, S, C, V));
        if (withIsotropicAnchored)
            best = std::max(best, anchoredTime(d0, u0, a0, H, T));
        const Vec3 w = centers.col(static_cast<Eigen::Index>(a)) - pb;
        const Vec3 n = w / w.norm();
        best = std::max(best, normalAnchoredTime(d0, n.dot(u0), n.dot(a0), H, T));
        return std::min(best, T);
    }
};

struct SelfPairStatic
{
    int f[Robot::nSelfPairs], g[Robot::nSelfPairs];
    std::size_t a[Robot::nSelfPairs], b[Robot::nSelfPairs];
    bool active[Robot::nSelfPairs];
    /// The same endpoints one byte wide. The screening sweep walks all 303 rows and
    /// touches five other 303-entry arrays on the way, so the 4.8 KB of `size_t`
    /// indices is cache the sweep cannot spare; 40 spheres fit in a byte.
    std::uint8_t ca[Robot::nSelfPairs], cb[Robot::nSelfPairs];
    int maxBaseFrame{0};
    SelfPairStatic()
    {
        for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
        {
            const auto &pr = Robot::selfPairs()[p];
            a[p] = pr.a; b[p] = pr.b;
            ca[p] = static_cast<std::uint8_t>(pr.a);
            cb[p] = static_cast<std::uint8_t>(pr.b);
            f[p] = static_cast<int>(Robot::spheres()[pr.a].frame);
            g[p] = static_cast<int>(Robot::spheres()[pr.b].frame);
            active[p] = g[p] > f[p];
            maxBaseFrame = std::max(maxBaseFrame, f[p]);
        }
    }
};
static const SelfPairStatic SP;

// (b) current path: L1 lever-arm polytope restricted to the ray, self rows only
static double repoSelfScale(const Robot::SphereCenters &centers, const Configuration &au,
                            double horizon, double selfMargin = 0.0)
{
    const auto &lever = Robot::selfPairLeverArms();
    const auto &radii = Robot::selfPairRadii();
    const auto &marg = Robot::selfPairMargins();
    double best = horizon;
    for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
    {
        const Eigen::Index ip = static_cast<Eigen::Index>(p);
        const double speed = lever.row(ip) * au;
        if (speed <= 0.0) continue;   // no moving joint affects this pair (safeScale's rule)
        const double slack =
            (centers.col(static_cast<Eigen::Index>(SP.a[p])) -
             centers.col(static_cast<Eigen::Index>(SP.b[p]))).norm() - radii[ip] - marg[ip]
            - selfMargin;
        if (slack <= 0.0) return 0.0;
        best = std::min(best, slack / speed);
    }
    return best;
}

// (c) hold-time path, same rows. Screened: every hold certificate is >= that
// pair's own L1 bound, so a pair whose L1 bound already exceeds the running
// best cannot bind and is skipped.
static double holdSelfScale(const Robot::Kinematics &kin, const Robot::SphereCenters &centers,
                            const Configuration &u, const Configuration &au, double horizon,
                            HoldCache &cache, bool withIso, long *evaluated,
                            double selfMargin = 0.0, bool alreadyBuilt = false)
{
    const auto &lever = Robot::selfPairLeverArms();
    const auto &radii = Robot::selfPairRadii();
    const auto &marg = Robot::selfPairMargins();

    double l1[Robot::nSelfPairs], slack[Robot::nSelfPairs];
    double l1min = horizon;
    std::size_t argmin = 0;
    for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
    {
        const Eigen::Index ip = static_cast<Eigen::Index>(p);
        const double speed = lever.row(ip) * au;
        if (speed <= 0.0) { l1[p] = horizon; slack[p] = 1.0; continue; }
        slack[p] = (centers.col(static_cast<Eigen::Index>(SP.a[p])) -
                    centers.col(static_cast<Eigen::Index>(SP.b[p]))).norm() - radii[ip] - marg[ip]
                   - selfMargin;
        if (slack[p] <= 0.0) return 0.0;
        l1[p] = slack[p] / speed;
        if (l1[p] < l1min) { l1min = l1[p]; argmin = p; }
    }

    if (!alreadyBuilt) cache.build(kin, u, horizon);

    long n = 0;
    double best = horizon;
    if (SP.active[argmin])
    {
        best = std::max(l1[argmin],
                        cache.pairHold(kin, centers, SP.a[argmin], SP.b[argmin], SP.f[argmin],
                                       SP.g[argmin], slack[argmin], horizon, withIso));
        ++n;
    }
    for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
    {
        if (p == argmin || !SP.active[p] || l1[p] >= best) continue;
        const double candidate =
            std::max(l1[p], cache.pairHold(kin, centers, SP.a[p], SP.b[p], SP.f[p], SP.g[p],
                                           slack[p], horizon, withIso));
        best = std::min(best, candidate);
        ++n;
    }
    *evaluated += n;
    return std::min(best, horizon);
}



// ---------------------------------------------------------------------------
// World rows. A sphere's certified region slack IS a displacement budget in
// metres, so the same hold time applies with d0 = slack_i and the chain based
// at frame 0 -- which the f=0 prefix cache already holds. No normal is used:
// the SDF is not convex, so the Cauchy-Schwarz step of the self-pair case is
// unavailable and the isotropic bound is what is left.
struct WorldStatic
{
    int frame[Robot::nSpheres];
    WorldStatic()
    {
        for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            frame[i] = static_cast<int>(Robot::spheres()[i].frame);
    }
};
static const WorldStatic WS;

template <typename TravelVector>
static double holdWorldScale(const Robot::Kinematics &kin, const Robot::SphereCenters &centers,
                             const Barrier::CertifiedRegion &region,
                             const TravelVector &travelUnit, double horizon,
                             const HoldCache &cache, long *evaluated)
{
    if (!region.valid) return 0.0;
    double l1[Robot::nSpheres];
    double l1min = horizon;
    std::size_t argmin = Robot::nSpheres;
    for (std::size_t i = 0; i < Robot::nSpheres; ++i)
    {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        const double rate = travelUnit[ii];
        if (rate <= 0.0 || WS.frame[i] == 0) { l1[i] = horizon; continue; }
        if (region.slack[ii] <= 0.0) return 0.0;
        l1[i] = region.slack[ii] / rate;
        if (l1[i] < l1min) { l1min = l1[i]; argmin = i; }
    }
    long n = 0;
    double best = horizon;
    auto one = [&](std::size_t i)
    {
        const int j = WS.frame[i] - 1;
        const Vec3 p = centers.col(static_cast<Eigen::Index>(i));
        const Vec3 last = p - kin.jointOrigin[j];
        const double L = last.norm();
        const double o = cache.Om[0][j], a2 = cache.A[0][j];
        double V = cache.cumV[0][j] + L * o;
        const double C = cache.cumC[0][j] + L * (a2 + o * o);
        const double H = cache.cumH[0][j] + L * (cache.W[0][j] + 3.0 * a2 * o + o * o * o);
        const Vec3 u0 = cache.cumU[0][j] + cache.omega[0][j].cross(last);
        const Vec3 a0 = cache.cumA[0][j] + cache.alpha[0][j].cross(last)
                        + cache.omega[0][j].cross(cache.omega[0][j].cross(last));
        const double S = u0.norm();
        V = std::max(V, S);
        const double d0 = region.slack[static_cast<Eigen::Index>(i)];
        double t = std::min(horizon, speedCappedTime(d0, S, C, V));
        t = std::max(t, anchoredTime(d0, u0, a0, H, horizon));
        t = std::max(t, std::min(horizon, l1[i]));   // never worse than the L1 bound
        ++n;
        return std::min(t, horizon);
    };
    if (argmin < Robot::nSpheres) best = one(argmin);
    for (std::size_t i = 0; i < Robot::nSpheres; ++i)
    {
        if (i == argmin || WS.frame[i] == 0 || l1[i] >= best) continue;
        best = std::min(best, one(i));
    }
    *evaluated += n;
    return std::min(best, horizon);
}


// ===========================================================================
// The fused query the rollout actually wants.
// ===========================================================================
//
// `holdSelfScale` and `holdWorldScale` above answer two questions the caller
// then takes a minimum of. The caller does not want two answers, and paying for
// two costs more than the second answer is worth:
//
//   - the geometry is computed three times over. `ClearanceBarrier::
//     certifiedRegion()` walks the chain, builds the 40 centres and measures all
//     303 pair distances; the caller then walks the chain and builds the centres
//     again for the certificate; `holdSelfScale` measures the 303 distances
//     again; and the diagnostic `repoSelfScale` measures them a third time.
//   - the two branch-and-bound searches do not see each other. Each starts its
//     running bound at the horizon, so a world row that has already pinned the
//     answer to 3 mm cannot stop the self search from evaluating pairs worth
//     20 mm, and vice versa.
//
// `HoldEngine` is the same certificate -- same spheres, same pair table, same
// SDF, same margins, same L1 rows, same speed-capped and normal-anchored terms,
// same clipping and the same min/max ordering -- computed once, with one shared
// bound. Everything it prunes is provably unable to lower the answer, because
// every row's certificate is at least that row's own L1 time.


// --------------------------------------------------- anchored roots, fast form
//
// Both anchored certificates bisect a monotone predicate, and both spent almost
// all of their time re-deriving, at every step, structure that is fixed for the
// row: the interior maximiser of the prefix envelope, and the square roots the
// comparison does not need.
//
//   - the maximiser is a property of (b, c) or of (u0, a0). It is found once.
//   - the predicate `sqrt(M(t)) + H t^3/6 <= d0` is equivalent, for the only t
//     where it can be true, to `M(t) <= (d0 - H t^3/6)^2` -- a squaring instead
//     of a square root, and none of the three vector norms the old form took per
//     step.
//
// The squared comparison carries a downward safety factor, so where rounding
// could make the two forms disagree the fast one refuses: the root it returns is
// never past the root the exact predicate would give. Certificate equation,
// envelope and clipping are untouched; only the arithmetic that finds the root
// of that equation is.
static constexpr double kRootSafety = 1.0 - 1e-14;

/// max_{0<=s<=t} (b s + c s^2/2) plus H t^3/6, against d0 -- with the interior
/// maximiser of the quadratic found once instead of once per bisection step.
struct NormalAnchor
{
    double b, c, H6, d0, sStar, pStar;

    NormalAnchor(double bb, double cc, double H, double dd)
      : b(bb), c(cc), H6(H * (1.0 / 6.0)), d0(dd), sStar(-1.0), pStar(0.0)
    {
        if (c < 0.0)
        {
            const double s = -b / c;
            // The vertex value -b^2/(2c) written as b*s/2, which needs no second
            // division.
            if (s > 0.0) { sStar = s; pStar = 0.5 * b * s; }
        }
    }

    double prefix(double t) const
    {
        double m = t * (b + 0.5 * c * t);
        if (m < 0.0) m = 0.0;
        if (sStar > 0.0 && sStar < t && pStar > m) m = pStar;
        return m;
    }

    /// Is the whole of [0, t] certified?
    bool holds(double t) const
    {
        HT_COUNT(rootIterations, 1);
        return prefix(t) + H6 * t * t * t <= d0 * kRootSafety;
    }

    double solve(double T) const
    {
        HT_COUNT(normalRoots, 1);
        if (holds(T)) return T;
        double lo = 0.0, hi = T;
        for (int i = 0; i < kBisect; ++i)
        {
            const double m = 0.5 * (lo + hi);
            if (holds(m)) lo = m; else hi = m;
        }
        return lo;
    }
};

/// max_{0<=s<=t} |s u0 + s^2 a0/2| plus H t^3/6, against d0, in squared form.
struct IsoAnchor
{
    double uu, ua, qa, H6, d0, rStar, pStar;

    IsoAnchor(const Vec3 &u0, const Vec3 &a0, double H, double dd)
      : uu(u0.squaredNorm())
      , ua(u0.dot(a0))
      , qa(0.25 * a0.squaredNorm())
      , H6(H * (1.0 / 6.0))
      , d0(dd)
      , rStar(-1.0)
      , pStar(0.0)
    {
        const double aa = 4.0 * qa;
        if (aa > 0.0)
        {
            const double disc = 9.0 * ua * ua - 8.0 * aa * uu;
            if (disc >= 0.0)
            {
                // psi' = s (aa s^2 + 3 ua s + 2 uu) with uu, aa >= 0: the smaller
                // root is the local maximum and the larger is the local minimum,
                // and a minimum can never set a prefix maximum. So one root, not
                // two, and one square root per row rather than one per step.
                const double r = (-3.0 * ua - std::sqrt(disc)) / (2.0 * aa);
                if (r > 0.0) { rStar = r; pStar = psi(r); }
            }
        }
    }

    /// |t u0 + t^2 a0/2|^2, in Horner form.
    double psi(double t) const { return t * t * (uu + t * (ua + t * qa)); }

    double reach2(double t) const
    {
        double m = psi(t);
        if (rStar > 0.0 && rStar < t && pStar > m) m = pStar;
        return m;
    }

    bool holds(double t) const
    {
        HT_COUNT(rootIterations, 1);
        const double rhs = d0 - H6 * t * t * t;
        if (rhs < 0.0)
            return false;
        return reach2(t) <= rhs * rhs * kRootSafety;
    }

    double solve(double T) const
    {
        HT_COUNT(isoRoots, 1);
        if (holds(T)) return T;
        double lo = 0.0, hi = T;
        for (int i = 0; i < kBisect; ++i)
        {
            const double m = 0.5 * (lo + hi);
            if (holds(m)) lo = m; else hi = m;
        }
        return lo;
    }
};

/// Everything the certificate asks of one (chain-base frame, sphere) endpoint. It
/// depends on the base frame and on that one sphere -- never on the sphere at the
/// other end of a pair.
///
/// Built in two halves, because most rows only need the first. `S`, `C` and `V`
/// settle the speed-capped term, and a row whose speed-capped term already exceeds
/// the running bound is finished: it never asks for `a0` or `H`, which is where the
/// second cross product and the third prefix sum live.
///
/// Several pair rows *can* share an endpoint, and memoizing them on `(frame,
/// sphere)` was tried: about 12 distinct endpoints serve about 13 rows per query, so
/// it saved roughly one build and cost a 240-slot table in L1. Measured, that was a
/// 10% loss. The endpoint is built on the stack instead.
struct Endpoint
{
    Vec3 last, u0, a0;
    double S, C, V, H;
};

/// One fused hold-time query, with every table it needs sized at compile time.
class HoldEngine
{
public:
    static constexpr int NS = static_cast<int>(Robot::nSpheres);
    static constexpr int NP = static_cast<int>(Robot::nSelfPairs);

    /// What the query found, for callers keeping diagnostics.
    struct Report
    {
        /// The pure-L1 answer over the self rows -- `repoSelfScale()`'s number. Filled
        /// only when a self row bound the answer, which is where the original measured
        /// it too; the screen no longer produces it for free, because it rejects most
        /// rows without ever taking their square root.
        double selfL1{0.0};
        /// Whether a self-pair row, rather than a world sphere row, set the answer.
        /// Only meaningful when `bounded`: with no row binding, neither family did.
        bool selfBinding{false};
        /// Whether a geometry row set the answer at all, rather than the horizon --
        /// or, on the screened path, the screening clip -- capping it. An unbounded
        /// query has no bottleneck to attribute: the certificate ran out of look-ahead,
        /// not out of clearance.
        bool bounded{false};
        /// Whether the answer was zero: some row's slack was already non-positive, so
        /// the control cannot be held at all. Blocked implies `bounded`, and
        /// `selfBinding` says which family it was.
        bool blocked{false};
        /// Input. `selfL1` costs a full 303-pair L1 sweep on top of the query, which is
        /// most of what the query saved; callers that only want the attribution clear
        /// it. Left on so the region rollout, which reports the gain over L1, is
        /// unaffected.
        bool wantSelfL1{true};
        /// Which term of the binding row's `max` achieved it. `None` when no row bound
        /// the answer, and also when the row that did was blocked outright -- a
        /// non-positive slack is rejected before any of the three terms is formed, so
        /// there is nothing to attribute.
        HoldTerm bindingTerm{HoldTerm::None};
        std::size_t selfEvaluated{0};   ///< self rows whose expensive certificate ran
        std::size_t worldEvaluated{0};  ///< world rows whose expensive certificate ran
    };

    explicit HoldEngine(const Barrier &barrier) : barrier_(barrier)
    {
        refreshMargin();
    }

    /// Reproduce the pre-optimization arithmetic exactly: the original bisection, no
    /// early exit on the anchored term. For validation only -- it is slower and no
    /// safer, and the rollout never asks for it.
    void setLegacyRoots(bool legacy) const { fastRoots_ = !legacy; }

    /// The certified arc length along `q + t u`, `t` in `[0, horizon]`: the smallest
    /// hold time over the 40 world sphere rows and the 303 self-pair rows, clipped to
    /// the horizon. Zero means blocked.
    double holdScale(const Configuration &q, const Configuration &u, double horizon) const
    {
        return query(q, u, horizon, nullptr);
    }

    double query(const Configuration &q, const Configuration &u, double horizon,
                 Report *report) const
    {
        barrier_.worldRegion(q, geo_);
        HT_COUNT(fk, 1);
        HT_COUNT(centers, 1);
        HT_COUNT(sdfBatch, 1);
        return queryAt(geo_.kin, geo_.centers, geo_.slack.data(), geo_.valid, u, horizon, report);
    }

    /// The same query against geometry the caller already has.
    ///
    /// A CBF-QP filter has just walked the chain, built the 40 sphere centres and
    /// queried the field to solve for its control; asking it how long that control may
    /// be held should not repeat any of that. `Evaluation` keeps what it built, so this
    /// takes it and starts at the screening pass.
    double holdScaleFrom(const Barrier::Evaluation &evaluation, const Configuration &u,
                         double horizon, Report *report = nullptr) const
    {
        barrier_.worldSlack(evaluation, borrowedSlack_);
        pairSubsetCount_ = -1;
        worldSubsetCount_ = -1;
        return queryAt(evaluation.kin, evaluation.centers, borrowedSlack_.data(),
                       evaluation.inBounds, u, horizon, report);
    }

    /// As above, sweeping only the self-collision pairs the caller's own screening kept.
    ///
    /// \p screenHorizon is the span that screening was done at: a pair it dropped has
    /// `h_ab > (bound on |dh_ab/dt| over the whole control box) * screenHorizon`, so it
    /// cannot reach zero within that span under *any* admissible control -- and the
    /// control being certified here is admissible. Clipping the answer to
    /// \p screenHorizon therefore makes the minimum over the kept pairs equal to the
    /// minimum over all 303, and the sweep collapses to the handful the QP was already
    /// looking at. In the rollout `screenHorizon` is `max(dt, 1/kappa)`, far longer than
    /// any hop, so the clip costs nothing.
    ///
    /// The 40 world rows are deliberately *not* restricted this way. Their slack is the
    /// smaller of clearance and distance-to-the-baked-box, and screening looks only at
    /// the first, so a row it dropped could still bind through the second. Forty rows
    /// are cheap; the 303 are what this is for.
    double holdScaleFrom(const Barrier::Evaluation &evaluation, const Configuration &u,
                         double horizon, double screenHorizon, Report *report = nullptr) const
    {
        // The world rows can join the restriction only when the baked box provably
        // contains the arm's reach. Otherwise their slack is the smaller of clearance
        // and distance-to-the-box, screening looks only at the first, and a row it
        // dropped could still bind through the second.
        const bool worldToo = barrier_.enclosesReach();
        int nw = 0, np = 0;
        for (int r = 0; r < evaluation.active; ++r)
        {
            const int constraint = evaluation.constraint[r];
            if (constraint >= NS)
                pairSubset_[np++] = constraint - NS;
            else if (worldToo)
                worldSubset_[nw++] = constraint;
        }

        // Nothing survived screening: no constraint of either family can reach zero
        // inside the screening horizon, so the whole of it is certified and there is
        // no geometry, no envelope and no row to look at.
        if (worldToo && nw == 0 && np == 0)
        {
            HT_COUNT(queries, 1);
            HT_COUNT(emptyActive, 1);
            ++emptyActiveTally();
            if (report != nullptr)
            {
                const bool wantSelfL1 = report->wantSelfL1;
                *report = Report();
                report->wantSelfL1 = wantSelfL1;
                report->selfL1 = std::min(horizon, screenHorizon);
            }
            return std::min(horizon, screenHorizon);
        }

        barrier_.worldSlack(evaluation, borrowedSlack_);
        pairSubsetCount_ = np;
        worldSubsetCount_ = worldToo ? nw : -1;
        const double span =
            queryAt(evaluation.kin, evaluation.centers, borrowedSlack_.data(),
                    evaluation.inBounds, u, std::min(horizon, screenHorizon), report);
        pairSubsetCount_ = -1;
        worldSubsetCount_ = -1;
        return span;
    }

    double queryAt(const Robot::Kinematics &kin, const Robot::SphereCenters &centres,
                   const double *worldSlack, bool valid, const Configuration &u, double horizon,
                   Report *report) const;

private:
    /// The envelope at (base frame f, sphere s), built where it is needed. It pulls
    /// the frame's prefix block into existence on first use.
    void endpoint(int f, int s, Endpoint &e) const
    {
        HT_COUNT(endpointEnvelopes, 1);
        cache_.ensureFrame(f);
        const int j = static_cast<int>(Robot::spheres()[s].frame) - 1;
        e.last = centers_->col(s) - kin_->jointOrigin[j];
        const double L = e.last.norm();
        const double o = cache_.Om[f][j], a2 = cache_.A[f][j];
        e.C = cache_.cumC[f][j] + L * (a2 + o * o);
        e.u0 = cache_.cumU[f][j] + cache_.omega[f][j].cross(e.last);
        e.S = e.u0.norm();
        e.V = std::max(cache_.cumV[f][j] + L * o, e.S);
    }

    /// The half only an anchored certificate needs.
    void anchorEndpoint(Endpoint &e, int f, int s) const
    {
        HT_COUNT(anchorEnvelopes, 1);
        const int j = static_cast<int>(Robot::spheres()[s].frame) - 1;
        const double L = e.last.norm();
        const double o = cache_.Om[f][j], a2 = cache_.A[f][j];
        e.H = cache_.cumH[f][j] + L * (cache_.W[f][j] + 3.0 * a2 * o + o * o * o);
        e.a0 = cache_.cumA[f][j] + cache_.alpha[f][j].cross(e.last) +
               cache_.omega[f][j].cross(cache_.omega[f][j].cross(e.last));
    }

    /// Pair row p's certified time, or any value at or above \p best when the row
    /// provably cannot lower the bound.
    ///
    /// The row is `max(L1, speed-capped, normal-anchored)`, and the first two are
    /// closed form. So the root solve -- the expensive part -- runs only when the
    /// cheap terms have failed to settle the row, and even then only after one
    /// evaluation of the anchored predicate at `best` has shown that the anchored
    /// term is not itself already past the bound. A root is computed only where its
    /// value actually moves the answer.
    double selfRow(int p, double T, double l1, double best) const
    {
        ++selfEvaluated_;
        HT_COUNT(selfExpensive, 1);
        const int f = SP.f[p], b = static_cast<int>(SP.b[p]);
        Endpoint e;
        endpoint(f, b, e);
        const double d0 = slack_[p];
        // Kept apart rather than folded into `t`, so the row can say which of them
        // answered it. Both are already clipped to the horizon, and `solve` never
        // returns more than its argument, so the three are directly comparable.
        termL1_ = std::min(T, l1);
        termSpeed_ = std::min(T, speedCappedTime(d0, e.S, e.C, e.V));
        termAnchor_ = 0.0;
        // The telescoping argument extends to a pair through its common ancestor frame,
        // but is not implemented here: self rows bind a minority of the time and their
        // own L1 already wins most of those, so there is little to recover.
        termLocal_ = 0.0;
        double t = std::max(termL1_, termSpeed_);
        if (fastRoots_ && t >= best)
            return t;
        anchorEndpoint(e, f, b);
        const Vec3 w = centers_->col(static_cast<Eigen::Index>(SP.a[p])) -
                       centers_->col(static_cast<Eigen::Index>(SP.b[p]));
        // The distance is already in hand from the screening pass; dividing by it is
        // the whole of what the normal costs here.
        const Vec3 n = w / dist_[p];
        const double bb = n.dot(e.u0), cc = n.dot(e.a0);
        if (!fastRoots_)
        {
            termAnchor_ = normalAnchoredTime(d0, bb, cc, e.H, T);
            return std::max(t, termAnchor_);
        }
        const NormalAnchor anchor(bb, cc, e.H, d0);
        if (anchor.holds(best))
        {
            // The row cannot lower the bound, so it will not be the binding row and
            // this value is never read for attribution.
            termAnchor_ = best;
            return best;   // the anchored term alone already exceeds the bound
        }
        termAnchor_ = anchor.solve(T);
        return std::max(t, termAnchor_);
    }

    /// World row i's certificate, same shape. No contact normal is available -- the
    /// SDF is not convex -- so the anchored term stays isotropic, exactly as before.
    double worldRow(int i, double T, double l1, double best) const
    {
        ++worldEvaluated_;
        HT_COUNT(worldExpensive, 1);
        ++localReachedTally();
        const double d0 = worldSlack_[i];
        termL1_ = std::min(T, l1);
        // Ordered before `endpoint()` deliberately. The local-lever term needs no
        // envelope, no `HoldCache` frame and no jerk bound -- six radii at the current
        // configuration -- so a row it settles is a row that costs nothing further.
        // Switchable so the term's contribution can be isolated; on by default. It is
        // never unsound to include (rho_ik(q0) <= leverArmBounds()(i,k)) and never
        // unsound to omit (the row falls back to the global L1 time).
        static const bool localEnabled = []
        {
            const char *v = std::getenv("OMPL_ENV_LOCAL");
            return v == nullptr || std::atoi(v) != 0;
        }();
        // A zero local rate means no joint that moves this sphere is moving, so the hold
        // is unbounded -- `T`, not zero. Zero is reserved for "this term is switched off",
        // where it must never win the `max`.
        const double localRate = localEnabled ? localWorldRate(i) : 0.0;
        termLocal_ = !localEnabled ? 0.0 : (localRate > 0.0 ? std::min(T, d0 / localRate) : T);
        termSpeed_ = 0.0;
        termAnchor_ = 0.0;
        double t = std::max(termL1_, termLocal_);
        if (fastRoots_ && t >= best)
        {
            if (localEnabled && termLocal_ >= best)
                ++localPrunedTally();
            return t;
        }
        Endpoint e;
        endpoint(0, i, e);
        termSpeed_ = std::min(T, speedCappedTime(d0, e.S, e.C, e.V));
        t = std::max(t, termSpeed_);
        if (fastRoots_ && t >= best)
            return t;
        anchorEndpoint(e, 0, i);
        if (!fastRoots_)
        {
            termAnchor_ = anchoredTime(d0, e.u0, e.a0, e.H, T);
            return std::max(t, termAnchor_);
        }
        const IsoAnchor anchor(e.u0, e.a0, e.H, d0);
        if (anchor.holds(best))
        {
            termAnchor_ = best;
            return best;
        }
        termAnchor_ = anchor.solve(T);
        return std::max(t, termAnchor_);
    }

    /// Telescoping local-lever rate for world sphere \p i: `sum_k rho_ik(q0) |u_k|`,
    /// with each `rho_ik` the *current* distance from the sphere centre to joint k's
    /// axis rather than its configuration-wide maximum.
    ///
    /// This is a certificate for the straight ray `q0 + t u`, not merely a linearisation,
    /// by a proximal-to-distal telescoping argument. Reach `q0 + t u` one joint at a
    /// time in index order. At step k, joints 1..k-1 have moved and joints k..n have
    /// not, so the change in the earlier joints is a *rigid* transform of the entire
    /// distal assembly -- joint k's axis and sphere i together, still in their original
    /// relative pose. A rigid transform preserves distance, so the radius the sphere
    /// swings on at step k is exactly `rho_ik(q0)`, and that step's displacement is
    /// `2 rho sin(|t u_k| / 2) <= rho_ik(q0) |t u_k|`. Summing with the triangle
    /// inequality bounds `|p_i(q0 + t u) - p_i(q0)|` by `t sum_k rho_ik(q0) |u_k|`.
    ///
    /// The bound is linear in `t`, hence monotone, so it also bounds the supremum over
    /// the whole prefix `[0, t]` -- which is what a hold certificate needs, and what a
    /// non-monotone bound would not give.
    ///
    /// Ordering is load-bearing: distal-to-proximal would move joints k+1..f(i) before
    /// step k, changing the sphere's distance to axis k, and the argument fails. And
    /// since `rho_ik(q0) <= max_q rho_ik(q) = leverArmBounds()(i, k)`, this is never
    /// weaker than the global L1 time it sits beside.
    ///
    /// `jointAxis` is a unit vector, so the perpendicular distance comes from two dot
    /// products rather than a cross product and a norm.
    double localWorldRate(int i) const
    {
        const Eigen::Vector3d p = centers_->col(i);
        const int frames = WS.frame[i];
        double rate = 0.0;
        for (int k = 0; k < frames; ++k)
        {
            const double speed = au_[k];
            if (speed <= 0.0)
                continue;
            const Eigen::Vector3d v = p - kin_->jointOrigin[k];
            const double axial = v.dot(kin_->jointAxis[k]);
            rate += speed * std::sqrt(std::max(v.squaredNorm() - axial * axial, 0.0));
        }
        return rate;
    }

    /// The cheapest term attaining the last fully evaluated row's `max`. Ties go to
    /// the cheaper term deliberately: `L1` here means the two expensive terms bought
    /// that row nothing, which is the question the counter exists to answer.
    HoldTerm classifyTerm() const
    {
        const double m =
            std::max(std::max(termL1_, termLocal_), std::max(termSpeed_, termAnchor_));
        if (termL1_ >= m)
            return HoldTerm::L1;
        if (termLocal_ >= m)
            return HoldTerm::LocalL1;
        if (termSpeed_ >= m)
            return HoldTerm::SpeedCapped;
        return HoldTerm::Anchored;
    }

    void refreshMargin() const
    {
        selfMargin_ = barrier_.selfMargin();
        const auto &radii = Robot::selfPairRadii();
        const auto &marg = Robot::selfPairMargins();
        for (int p = 0; p < NP; ++p)
            offset_[p] = radii[p] + marg[p] + selfMargin_;
    }

    const Barrier &barrier_;
    mutable bool fastRoots_{true};
    mutable double selfMargin_{0.0};
    mutable double offset_[NP];
    mutable Barrier::WorldRegion geo_;   ///< scratch for the self-computing path
    mutable Eigen::Matrix<double, NS, 1> borrowedSlack_;
    mutable const Robot::Kinematics *kin_{nullptr};
    mutable const Robot::SphereCenters *centers_{nullptr};
    mutable const double *worldSlack_{nullptr};
    mutable int pairSubset_[NP];
    mutable int pairSubsetCount_{-1};   ///< -1 sweeps all 303
    mutable int worldSubset_[NS];
    mutable int worldSubsetCount_{-1};  ///< -1 sweeps all 40
    mutable HoldCache cache_;
    mutable Eigen::Matrix<double, NS, 1> worldRate_;
    mutable Eigen::Matrix<double, NP, 1> selfSpeed_;
    mutable double slack_[NP];
    mutable double dist_[NP];
    mutable std::size_t selfEvaluated_{0};
    mutable std::size_t worldEvaluated_{0};
    /// The last fully evaluated row's three terms, for `classifyTerm()`. Written by
    /// every `selfRow`/`worldRow` call and read only for the row that binds.
    mutable double termL1_{0.0}, termSpeed_{0.0}, termAnchor_{0.0}, termLocal_{0.0};
    mutable Configuration au_{Configuration::Zero()};   ///< |u|, for `localWorldRate`
};

inline double HoldEngine::queryAt(const Robot::Kinematics &kin,
                                  const Robot::SphereCenters &centres, const double *worldSlack,
                                  bool valid, const Configuration &u, double horizon,
                                  Report *report) const
{
    HT_COUNT(queries, 1);
    if (selfMargin_ != barrier_.selfMargin())
        refreshMargin();
    selfEvaluated_ = 0;
    worldEvaluated_ = 0;
    kin_ = &kin;
    centers_ = &centres;
    worldSlack_ = worldSlack;

    if (report != nullptr)
    {
        report->selfL1 = horizon;
        report->selfBinding = false;
        report->bounded = false;
        report->blocked = false;
        report->bindingTerm = HoldTerm::None;
        report->selfEvaluated = 0;
        report->worldEvaluated = 0;
    }
    if (!valid)
    {
        HT_COUNT(zeroReturns, 1);
        // Out of the baked box is a world statement, not a self-collision one.
        if (report != nullptr)
        {
            report->bounded = true;
            report->blocked = true;
        }
        return 0.0;   // holdWorldScale()'s answer for an out-of-box region
    }

    const Configuration au = u.cwiseAbs();
    au_ = au;
    if (worldSubsetCount_ < 0)
        worldRate_.noalias() = Robot::leverArmBounds() * au;
    else
        for (int wk = 0; wk < worldSubsetCount_; ++wk)
            worldRate_[worldSubset_[wk]] = Robot::leverArmBounds().row(worldSubset_[wk]) * au;

    // ------------------------------------------------------- world rows, L1 first
    //
    // The 40 world rows are screened before the 303 pair rows, and not for symmetry:
    // one of them is the globally tightest row about seven times in ten, and its
    // certificate is what gives the pair sweep a bound worth screening against. A
    // minimum of ratios needs no ratio per row -- `slack/rate` can only lower the
    // running best where `slack < rate * best` -- so this costs 40 multiplies and a
    // handful of divisions.
    double worldMin = horizon;
    int worldArg = -1;
    const int worldSweep = worldSubsetCount_ < 0 ? NS : worldSubsetCount_;
    for (int wk = 0; wk < worldSweep; ++wk)
    {
        const int i = worldSubsetCount_ < 0 ? wk : worldSubset_[wk];
        const double rate = worldRate_[i];
        if (rate <= 0.0 || WS.frame[i] == 0)
            continue;
        HT_COUNT(worldRows, 1);
        const double slack = worldSlack[i];
        if (slack <= 0.0)
        {
            HT_COUNT(zeroReturns, 1);
            if (report != nullptr)
            {
                report->bounded = true;
                report->blocked = true;
            }
            return 0.0;
        }
        if (slack < rate * worldMin)
        {
            worldMin = slack / rate;
            worldArg = i;
        }
    }

    double best = horizon;
    bool selfBound = false;
    // The term that produced `best`, tracked alongside the family that did. Captured
    // at each point `best` falls, because `classifyTerm()` reads scratch that the very
    // next row overwrites.
    HoldTerm bindingTerm = HoldTerm::None;
    if (worldArg >= 0)
    {
        // The envelope -- the one genuinely superlinear piece of this query -- is
        // built here and only here, the first time a row actually asks for it.
        cache_.buildLazy(*kin_, u, horizon);
        HT_COUNT(cacheBuilds, 1);
        HT_COUNT(seedWorld, 1);
        best = worldRow(worldArg, horizon, worldMin, horizon);
        if (best < horizon)
            bindingTerm = classifyTerm();
    }

    // ---------------------------------------------------- self pairs, screened hard
    //
    // With a bound in hand the pair sweep needs neither a square root nor a division
    // for the rows it is going to reject, and it rejects almost all of them. Pair p
    // can only lower the bound if
    //
    //     |p_a - p_b|  <  radii + margins + speed * best
    //
    // and both sides are non-negative, so the test is a comparison of squares: one
    // multiply-add, one multiply, one branch. A row that fails it is *also* proved
    // clear -- the right-hand side exceeds the radii sum outright -- so nothing that
    // could block the motion escapes the screen.
    constexpr double kScreenSlack = 1.0 + 4e-16;
    struct Candidate
    {
        double l1;
        int row;   // p for self pair p, NP + i for world sphere i
    };
    Candidate candidates[NP + NS];
    int n = 0;

    // One 303 x 6 contraction is cheaper than 303 strided row dots, but not cheaper
    // than a handful of them: with a subset in hand, only those rows are contracted.
    if (pairSubsetCount_ < 0)
        selfSpeed_.noalias() = Robot::selfPairLeverArms() * au;
    else
        for (int k = 0; k < pairSubsetCount_; ++k)
            selfSpeed_[pairSubset_[k]] = Robot::selfPairLeverArms().row(pairSubset_[k]) * au;
    const Robot::SphereCenters &centers = centres;
    const auto &radii = Robot::selfPairRadii();
    const auto &marg = Robot::selfPairMargins();
    const double reach = best * kScreenSlack;
    const int sweep = pairSubsetCount_ < 0 ? NP : pairSubsetCount_;
    for (int k = 0; k < sweep; ++k)
    {
        const int p = pairSubsetCount_ < 0 ? k : pairSubset_[k];
        const double speed = selfSpeed_[p];
        if (speed <= 0.0)
            continue;
        HT_COUNT(selfDistances, 1);
        const double d2 = (centers.col(static_cast<Eigen::Index>(SP.ca[p])) -
                           centers.col(static_cast<Eigen::Index>(SP.cb[p])))
                              .squaredNorm();
        // A non-positive bound means the radii-plus-margin offset is itself negative
        // -- the benchmark's "pair rows off" setting drives it to -1e6 -- and then no
        // separation can be short enough to matter. Squaring would lose that sign.
        const double bound = offset_[p] + speed * reach;
        if (bound <= 0.0 || d2 >= bound * bound)
            continue;
        HT_COUNT(selfSqrt, 1);
        const double d = std::sqrt(d2);
        const double slack = d - radii[p] - marg[p] - selfMargin_;
        if (slack <= 0.0)
        {
            HT_COUNT(zeroReturns, 1);
            if (report != nullptr)
            {
                report->selfBinding = true;
                report->bounded = true;
                report->blocked = true;
            }
            return 0.0;
        }
        const double l1 = slack / speed;
        if (l1 < best)
        {
            dist_[p] = d;
            slack_[p] = slack;
            candidates[n++] = {l1, p};
        }
    }
    HT_COUNT(selfCandidates, static_cast<std::size_t>(n));
    [[maybe_unused]] const int selfCandidateCount = n;

    // The remaining world rows, under the same rule.
    for (int wk = 0; wk < worldSweep; ++wk)
    {
        const int i = worldSubsetCount_ < 0 ? wk : worldSubset_[wk];
        const double rate = worldRate_[i];
        if (rate <= 0.0 || WS.frame[i] == 0 || i == worldArg)
            continue;
        if (worldSlack[i] >= rate * reach)
            continue;
        const double l1 = worldSlack[i] / rate;
        if (l1 < best)
            candidates[n++] = {l1, NP + i};
    }
    HT_COUNT(worldCandidates, static_cast<std::size_t>(n - selfCandidateCount));
    HT_COUNT(candidates, static_cast<std::size_t>(n));

    if (n > 0 && worldArg < 0)
    {
        cache_.buildLazy(*kin_, u, horizon);
        HT_COUNT(cacheBuilds, 1);
    }
    if (n == 0 && worldArg < 0)
        HT_COUNT(noSeed, 1);

    // Rows are worth evaluating in increasing L1 order, and only until the next one
    // cannot beat the bound. Index order is the wrong order: a self pair worth 20 mm
    // evaluated before the world sphere worth 3 mm pays for a certificate that was
    // never going to bind. The candidate set is what survives the seed -- a couple of
    // dozen of 343 rows -- so ordering it costs a sort over that, not over the table.
    std::sort(candidates, candidates + n,
              [](const Candidate &x, const Candidate &y) { return x.l1 < y.l1; });

    for (int k = 0; k < n; ++k)
    {
        // Every remaining row has an L1 at least this one's, and every row's
        // certificate is at least its own L1, so nothing left can lower the bound.
        if (candidates[k].l1 >= best)
            break;
        const int row = candidates[k].row;
        if (row < NP)
        {
            const double value = std::min(horizon, selfRow(row, horizon, candidates[k].l1, best));
            if (value < best)
            {
                best = value;
                selfBound = true;
                bindingTerm = classifyTerm();
            }
        }
        else
        {
            const int i = row - NP;
            const double value = worldRow(i, horizon, candidates[k].l1, best);
            if (value < best)
            {
                best = value;
                selfBound = false;
                bindingTerm = classifyTerm();
            }
        }
    }

    if (report != nullptr)
    {
        report->selfBinding = selfBound;
        // `best` starts at the horizon and only ever falls, and every row clips its own
        // certificate there, so `best < horizon` holds exactly when some row -- rather
        // than the look-ahead the caller asked for -- set the answer. That is the
        // condition under which `selfBinding` names a bottleneck at all: a query that
        // ran out of horizon has no binding row, and the seed row leaves `selfBound`
        // false whether it bound the answer or returned the horizon untouched.
        report->bounded = best < horizon;
        report->bindingTerm = report->bounded ? bindingTerm : HoldTerm::None;
        report->selfEvaluated = selfEvaluated_;
        report->worldEvaluated = worldEvaluated_;
        // The pure-L1 answer over the pair rows, which the screen no longer produces
        // for free: it rejects most rows without ever taking their square root. It is
        // a diagnostic and only a self-binding step reports it, so it is measured
        // here, where the original measured it, and at the same rate.
        if (selfBound && report->wantSelfL1)
            report->selfL1 = repoSelfScale(centers, au, horizon, selfMargin_);
    }
    return std::min(best, horizon);
}

}  // namespace holdtime
