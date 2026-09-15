#!/usr/bin/env python3
"""Publication figure comparing two certificates on one planar-arm motion."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "certificate_horizon_explainer"
L = np.array([0.90, 0.70])
R_ARM = 0.035
Q0 = np.array([0.35, 0.85])
U = np.array([0.70, -0.45])
T_COLLISION = 1.42
R_OBS = 0.15

INK = "#202a2e"
GRAY = "#6f7b80"
LIGHT = "#d9ddde"
FIELD = "#ecebea"
BLUE = "#326b80"
ORANGE = "#c87528"
COLLISION = "#b8b5b2"


def fk(q):
    theta = np.cumsum(np.asarray(q), axis=-1)
    links = L * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return np.vstack((np.zeros(2), np.cumsum(links, axis=0)))


OBSTACLE = fk(Q0 + T_COLLISION * U)[-1]


def nearest_on_segment(x, a, b):
    d = b - a
    alpha = np.clip(np.dot(x - a, d) / np.dot(d, d), 0.0, 1.0)
    p = a + alpha * d
    return np.linalg.norm(x - p), p


def clearance(q):
    # This explanatory row is the spherical end-effector marker. Its exact
    # configuration obstacle is therefore ||p_i(q)-o|| <= r_i+r_o.
    return np.linalg.norm(fk(q)[-1] - OBSTACLE) - R_OBS - R_ARM


def capsule(ax, a, b, radius, face, edge, alpha=1.0, zorder=2):
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = radius * np.array((-tangent[1], tangent[0]))
    ax.add_patch(Polygon((a + normal, b + normal, b - normal, a - normal),
                         closed=True, facecolor=face, edgecolor=edge,
                         linewidth=0.55, alpha=alpha, zorder=zorder))
    for center in (a, b):
        ax.add_patch(Circle(center, radius, facecolor=face, edgecolor=edge,
                            linewidth=0.55, alpha=alpha, zorder=zorder))


def arm(ax, q, face, edge, alpha=1.0, zorder=2):
    p = fk(q)
    for a, b in zip(p[:-1], p[1:]):
        capsule(ax, a, b, R_ARM, face, edge, alpha, zorder)
    ax.scatter(p[:, 0], p[:, 1], s=(22, 16, 13), facecolor=face, edgecolor=edge,
               linewidth=0.55, alpha=alpha, zorder=zorder + 1)


def motion_constants(u):
    """Constants in the endpoint speed envelope for a held velocity u."""
    # Motion-dependent speed envelope.  For a planar two-link endpoint,
    # ||p_dot(0)|| is exact, C bounds ||p_ddot||, and V is the triangle bound
    # on speed.  Integrating min(S + Ct, V) gives a prefix displacement bound.
    th = np.cumsum(Q0)
    j0 = np.column_stack((
        L[0] * np.array((-np.sin(th[0]), np.cos(th[0])))
        + L[1] * np.array((-np.sin(th[1]), np.cos(th[1]))),
        L[1] * np.array((-np.sin(th[1]), np.cos(th[1]))),
    ))
    s0 = np.linalg.norm(j0 @ u)
    c = L[0] * u[0] ** 2 + L[1] * (u.sum()) ** 2
    v = L[0] * abs(u[0]) + L[1] * abs(u.sum())
    tc = max(0.0, (v - s0) / c)
    return s0, c, v, tc


def displacement_bounds(t, u=U):
    """Actual endpoint displacement and two analytic bounds for this arm."""
    p0 = fk(Q0)[-1]
    actual = np.array([np.linalg.norm(fk(Q0 + s * u)[-1] - p0) for s in t])
    a_rate = L.sum() * abs(u[0]) + L[1] * abs(u[1])
    d_lip = a_rate * t
    s0, c, v, tc = motion_constants(u)
    d_tc = s0 * tc + 0.5 * c * tc ** 2
    d_disp = np.where(t <= tc, s0 * t + 0.5 * c * t ** 2,
                      d_tc + v * (t - tc))
    return actual, d_lip, d_disp, dict(A=a_rate, S=s0, C=c, V=v, cap_time=tc)


def motion_time(clearance_value, u):
    """Invert the integrated speed envelope analytically."""
    s0, c, v, tc = motion_constants(u)
    d_tc = s0 * tc + 0.5 * c * tc ** 2
    if clearance_value <= d_tc:
        return (-s0 + np.sqrt(s0 ** 2 + 2.0 * c * clearance_value)) / c
    return tc + (clearance_value - d_tc) / v


def certified_regions(clearance_value, samples=721):
    """Radial boundaries using held controls with ||u||=||U||."""
    theta = np.linspace(-np.pi, np.pi, samples)
    speed = np.linalg.norm(U)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    controls = speed * directions
    lip = []
    disp = []
    for d, u in zip(directions, controls):
        a_rate = L.sum() * abs(u[0]) + L[1] * abs(u[1])
        lip.append(Q0 + d * speed * clearance_value / a_rate)
        disp.append(Q0 + d * speed * motion_time(clearance_value, u))
    return np.asarray(lip), np.asarray(disp)


def first_crossing(t, y, level):
    k = np.flatnonzero(y >= level)[0]
    return float(np.interp(level, y[k - 1:k + 1], t[k - 1:k + 1]))


def collision_grid(q1, q2):
    z = np.empty((q2.size, q1.size), dtype=bool)
    for iy, b in enumerate(q2):
        for ix, a in enumerate(q1):
            z[iy, ix] = clearance((a, b)) <= 0.0
    return z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    h0 = clearance(Q0)
    t = np.linspace(0.0, 1.55, 500)
    actual, d_lip, d_disp, constants = displacement_bounds(t)
    tau_lip = first_crossing(t, d_lip, h0)
    tau_disp = first_crossing(t, d_disp, h0)

    fig = plt.figure(figsize=(7.60, 2.36))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1), wspace=0.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Workspace geometry. Pale poses establish the actual sweep without implying regions.
    ax = axes[0]
    for s, alpha in zip(np.linspace(0.18, tau_disp, 5), np.linspace(0.10, 0.22, 5)):
        arm(ax, Q0 + s * U, LIGHT, LIGHT, alpha=alpha, zorder=1)
    ee = np.array([fk(Q0 + s * U)[-1] for s in np.linspace(0, tau_disp, 100)])
    ax.plot(ee[:, 0], ee[:, 1], color=BLUE, linewidth=0.8, linestyle=(0, (2, 2)), zorder=2)
    ax.add_patch(Circle(OBSTACLE, R_OBS, facecolor=FIELD, edgecolor=INK,
                        linewidth=0.8, zorder=2))
    arm(ax, Q0, "#f7f7f6", INK, zorder=3)
    p = fk(Q0)
    nearest = min((nearest_on_segment(OBSTACLE, a, b) for a, b in zip(p[:-1], p[1:])),
                  key=lambda item: item[0])[1]
    n = (OBSTACLE - nearest) / np.linalg.norm(OBSTACLE - nearest)
    a, b = nearest + R_ARM * n, OBSTACLE - R_OBS * n
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="|-|", color=ORANGE,
                                                     linewidth=0.8))
    ax.annotate(r"$h_i(q_0)$", xy=0.5 * (a + b), xytext=(-9, 21),
                textcoords="offset points", color=ORANGE, ha="center", va="bottom")
    u_anchor = ee[22] + np.array((-0.025, 0.055))
    ax.annotate(r"$u^\star$", xy=u_anchor, xytext=(18, 18), textcoords="offset points",
                color=BLUE, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=0.7))
    ax.annotate(r"$q_0$", xy=p[1], xytext=(-7, 8), textcoords="offset points",
                ha="right", va="bottom")
    ax.set(xlim=(-0.08, 1.67), ylim=(-0.08, 1.67), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Actual radial certificate boundaries, with the row obstacle behind them.
    ax = axes[1]
    q1 = np.linspace(-1.15, 1.85, 420)
    q2 = np.linspace(-0.65, 2.35, 420)
    z = collision_grid(q1, q2)
    ax.contourf(q1, q2, z.astype(float), levels=(0.5, 1.5), colors=(COLLISION,), alpha=0.45)
    ax.contour(q1, q2, z.astype(float), levels=(0.5,), colors=(GRAY,), linewidths=0.7)
    lip_region, disp_region = certified_regions(h0)
    ax.fill(disp_region[:, 0], disp_region[:, 1], facecolor=BLUE, alpha=0.10,
            edgecolor=BLUE, linewidth=1.0, zorder=2)
    ax.fill(lip_region[:, 0], lip_region[:, 1], facecolor=ORANGE, alpha=0.16,
            edgecolor=ORANGE, linewidth=1.0, zorder=3)
    ray_t = np.linspace(0, 1.52, 200)
    ray = Q0 + ray_t[:, None] * U
    ax.plot(ray[:, 0], ray[:, 1], color=GRAY, linewidth=0.75, zorder=2)
    q_lip, q_disp = Q0 + tau_lip * U, Q0 + tau_disp * U
    ax.plot((Q0[0], q_lip[0]), (Q0[1], q_lip[1]), color=ORANGE,
            linewidth=2.0, solid_capstyle="butt", zorder=4)
    ax.plot((q_lip[0], q_disp[0]), (q_lip[1], q_disp[1]), color=BLUE,
            linewidth=2.0, solid_capstyle="butt", zorder=4)
    ax.scatter((Q0[0], q_lip[0], q_disp[0]), (Q0[1], q_lip[1], q_disp[1]),
               s=(18, 16, 16), facecolor=(INK, ORANGE, BLUE), edgecolor="white",
               linewidth=0.45, zorder=5)
    ax.annotate(r"$q_0$", xy=Q0, xytext=(-7, 8), textcoords="offset points",
                ha="right", va="bottom")
    ax.annotate(r"$q_{\mathrm{lip}}$", xy=q_lip, xytext=(-8, -14),
                textcoords="offset points", color=ORANGE, ha="right", va="top")
    ax.annotate(r"$q_{\mathrm{disp}}$", xy=q_disp, xytext=(8, 8),
                textcoords="offset points", color=BLUE, ha="left", va="bottom")
    ax.text(-1.02, 2.18, r"$\mathcal{C}_{\mathrm{disp}}(q_0)$", color=BLUE, ha="left")
    ax.text(-1.02, 1.94, r"$\mathcal{C}_{\mathrm{lip}}(q_0)$", color=ORANGE, ha="left")
    ax.text(1.29, 1.43, r"$\mathcal{C}_{\mathrm{obs}}$", color=GRAY, ha="center")
    ax.set(xlabel=r"$q_1$ (rad)", ylabel=r"$q_2$ (rad)",
           xlim=(q1[0], q1[-1]), ylim=(q2[0], q2[-1]), aspect="equal")
    ax.spines[["top", "right"]].set_visible(False)

    # The two certificate bounds for the held-control ray.
    ax = axes[2]
    ax.plot(t, d_lip, color=ORANGE, linewidth=1.45, label=r"Lipschitz")
    ax.plot(t, d_disp, color=BLUE, linewidth=1.45, label=r"Kinematic")
    ax.axhline(h0, color=GRAY, linewidth=0.7, linestyle=(0, (2, 2)))
    for tau, color in ((tau_lip, ORANGE), (tau_disp, BLUE)):
        ax.plot((tau, tau), (0, h0), color=color, linewidth=0.65, linestyle=(0, (1, 2)))
        ax.scatter(tau, h0, s=15, facecolor=color, edgecolor="white", linewidth=.4, zorder=4)
    ax.text(0.05, h0 + .035, r"$h_i(q_0)$", color=GRAY)
    ax.annotate(r"$\tau_{\mathrm{lip}}$", xy=(tau_lip, h0), xytext=(-7, 8),
                textcoords="offset points", color=ORANGE, ha="right", va="bottom")
    # Put each label in the empty upper-left quadrant of its crossing so the
    # rising bound does not run through the text.
    ax.annotate(r"$\tau_{\mathrm{disp}}$", xy=(tau_disp, h0), xytext=(0, 9),
                textcoords="offset points", color=BLUE, ha="right", va="bottom")
    ax.set(xlabel=r"Hold Time $t$", ylabel="Displacement",
           xlim=(0, 1.52), ylim=(0, max(h0 * 1.45, 1.0)))
    ax.set_box_aspect(1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.60, 1.015), ncol=2,
              frameon=False, fontsize=7.4, handlelength=1.8,
              handletextpad=.4, columnspacing=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # Figure coordinates keep the three panel labels on one horizontal baseline;
    # axis-relative coordinates would stagger them because the panels have
    # different aspect ratios.
    fig.canvas.draw()
    label_y = 0.955
    for label, ax in zip(("a", "b", "c"), axes):
        fig.text(ax.get_position().x0 - 0.018, label_y, f"({label})",
                 fontsize=9.2, fontweight="bold", ha="left", va="top")

    fig.savefig(OUT / "certificate_horizons.pdf", bbox_inches="tight")
    fig.savefig(OUT / "certificate_horizons.png", dpi=300, bbox_inches="tight")
    audit = {
        "q0_rad": Q0.tolist(), "held_control_rad_per_s": U.tolist(),
        "link_lengths_m": L.tolist(), "arm_radius_m": R_ARM,
        "obstacle_center_m": OBSTACLE.tolist(), "obstacle_radius_m": R_OBS,
        "initial_clearance_m": h0, "tau_lip_s": tau_lip, "tau_disp_s": tau_disp,
        "motion_bound_constants": constants,
        "certified_regions": "radial boundaries evaluated for 721 held-control directions",
        "collision_region": "exact end-effector-sphere obstacle sampled on the displayed grid",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
