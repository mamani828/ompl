#pragma once
// Hold-time self-pair certificate, extracted from holdbench.cpp (validated there
// against exact FK: worst certified - true contact = -2.81e-04 over 400 configs).
#include <ompl/cbf/ClearanceBarrier.h>
#include <ompl/robots/UR5.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace holdtime
{
using Robot = ompl::robots::UR5;
using Barrier = ompl::cbf::ClearanceBarrier;
using Configuration = Robot::Configuration;
using Vec3 = Eigen::Vector3d;
static constexpr int NJ = 6;

// ---------------------------------------------------------------- scalar roots

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
    if (prefixMaxNorm(T, u, a) + H * T * T * T / 6.0 <= d0) return T;
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
    if (prefixMaxQuad(T, b, c) + H * T * T * T / 6.0 <= d0) return T;
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

    // One O(n^2) pass. Pairwise cosines are frame-invariant, so this serves
    // EVERY frame window -- all 303 pairs read out of it.
    void build(const Robot::Kinematics &kin, const Configuration &v, double T)
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
                    const double cw = std::cos(w), sw = std::sin(w);
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

        // Prefix quantities depend only on the BASE frame f, so six of these
        // cover all 21 possible frame windows.
        for (int f = 0; f < NJ; ++f)
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
    SelfPairStatic()
    {
        for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
        {
            const auto &pr = Robot::selfPairs()[p];
            a[p] = pr.a; b[p] = pr.b;
            f[p] = static_cast<int>(Robot::spheres()[pr.a].frame);
            g[p] = static_cast<int>(Robot::spheres()[pr.b].frame);
            active[p] = g[p] > f[p];
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

static double holdWorldScale(const Robot::Kinematics &kin, const Robot::SphereCenters &centers,
                             const Barrier::CertifiedRegion &region,
                             const Barrier::Values &travelUnit, double horizon,
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

}  // namespace holdtime
