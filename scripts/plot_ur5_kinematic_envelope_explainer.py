#!/usr/bin/env python3
"""Computed UR5 illustration of the Level-2 kinematic hold certificate.

All geometry comes from the spherized UR5 URDF.  The robot follows the exact
ray q(t)=q0+t*u; the obstacle placement, ghost poses, clearances, Jacobians,
and envelope curves are numerical outputs, not illustration control points.
"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hold_time import Geometry, level2, speed_capped_bound  # noqa: E402

URDF = ROOT / "external/vamp/resources/ur5/ur5_spherized.urdf"
OUT = ROOT / "results/ur5_kinematic_envelope_explainer"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def vec(element, key="xyz"):
    return np.fromstring(element.get(key, "0 0 0"), sep=" ") if element is not None else np.zeros(3)


class UR5:
    def __init__(self):
        root = ET.parse(URDF).getroot()
        children = {j.find("child").get("link") for j in root.findall("joint")}
        links = {x.get("name") for x in root.findall("link")}
        roots = sorted(links - children)
        if len(roots) != 1:
            raise RuntimeError(f"expected one URDF root, got {roots}")
        self.root = roots[0]
        pending = list(root.findall("joint")); known = {self.root}; self.chain = []
        self.parent_joint = {}
        while pending:
            progressed = False
            for j in pending[:]:
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                if parent not in known:
                    continue
                origin = j.find("origin"); T = np.eye(4)
                T[:3, 3] = vec(origin)
                T[:3, :3] = Rotation.from_euler("xyz", vec(origin, "rpy")).as_matrix()
                idx = JOINTS.index(j.get("name")) if j.get("name") in JOINTS else -1
                record = (parent, child, T, idx, vec(j.find("axis")), j.get("name"))
                self.chain.append(record); self.parent_joint[child] = record
                known.add(child); pending.remove(j); progressed = True
            if not progressed:
                raise RuntimeError("could not topologically order URDF")
        self.spheres = []
        for link in root.findall("link"):
            for col in link.findall("collision"):
                sphere = col.find("geometry/sphere")
                if sphere is not None:
                    self.spheres.append((link.get("name"), vec(col.find("origin")),
                                         float(sphere.get("radius"))))

    def fk(self, q):
        transforms = {self.root: np.eye(4)}; joint_frames = {}
        for parent, child, T, idx, axis, name in self.chain:
            frame = transforms[parent] @ T
            if idx >= 0:
                joint_frames[idx] = (frame[:3, 3].copy(), frame[:3, :3] @ axis)
                R = np.eye(4); R[:3, :3] = Rotation.from_rotvec(axis*q[idx]).as_matrix()
                frame = frame @ R
            transforms[child] = frame
        centers = np.array([(transforms[link] @ np.r_[local, 1])[:3]
                            for link, local, _ in self.spheres])
        return transforms, joint_frames, centers

    def influence(self, sphere_index, transforms, joint_frames, center):
        link = self.spheres[sphere_index][0]; indices = []
        while link != self.root:
            rec = self.parent_joint[link]
            if rec[3] >= 0:
                indices.append(rec[3])
            link = rec[0]
        indices.reverse()
        if not indices:
            return np.array([], dtype=int), None
        origins = [joint_frames[i][0] for i in indices]
        axes = [joint_frames[i][1] / np.linalg.norm(joint_frames[i][1]) for i in indices]
        segments = [origins[k+1] - origins[k] for k in range(len(origins)-1)]
        segments.append(center - origins[-1])
        return np.array(indices), Geometry(np.array(axes), np.array(segments))


def sphere_mesh(center, radius, n=28):
    a = np.linspace(0, 2*np.pi, n); b = np.linspace(0, np.pi, n//2)
    return (center[0] + radius*np.outer(np.cos(a), np.sin(b)),
            center[1] + radius*np.outer(np.sin(a), np.sin(b)),
            center[2] + radius*np.outer(np.ones_like(a), np.cos(b)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = UR5()
    q0 = np.array([-1.10, -1.02, 1.28, -1.38, -0.92, 0.48])
    u = np.array([0.48, -0.36, 0.56, 0.30, -0.34, 0.24])
    horizon, target_time, obstacle_radius = 1.60, 1.20, 0.065
    _, _, c0 = robot.fk(q0); _, _, chit = robot.fk(q0 + target_time*u)
    radii = np.array([x[2] for x in robot.spheres])
    # Derive the obstacle from the distal sphere with the greatest exact-FK travel.
    distal = [i for i, (link, _, _) in enumerate(robot.spheres)
              if any(s in link for s in ("wrist_3", "robotiq", "finger"))]
    target = max(distal, key=lambda i: np.linalg.norm(chit[i]-c0[i]))
    obstacle = chit[target].copy()
    h0 = np.linalg.norm(c0-obstacle, axis=1) - radii - obstacle_radius
    if h0.min() <= 0:
        raise RuntimeError("derived obstacle is not initially separated")

    transforms0, frames0, _ = robot.fk(q0)
    rows = []
    for i in range(len(c0)):
        indices, geometry = robot.influence(i, transforms0, frames0, c0[i])
        if geometry is None:
            rows.append((horizon, indices, geometry, None))
            continue
        result = level2(geometry, u[indices], h0[i], horizon)
        rows.append((result.tau, indices, geometry, result))
    binding = int(np.argmin([x[0] for x in rows])); tau = rows[binding][0]

    times = np.linspace(0, horizon, 801)
    centers = np.array([robot.fk(q0+t*u)[2] for t in times])
    clearance = np.linalg.norm(centers-obstacle, axis=2) - radii[None, :] - obstacle_radius
    actual_min = clearance.min(axis=1)
    certified = np.empty((len(times), len(rows)))
    displacement = np.linalg.norm(centers-c0[None, :, :], axis=2)
    bounds = np.empty_like(certified)
    for i, (_, indices, geometry, result) in enumerate(rows):
        if geometry is None:
            bounds[:, i] = 0.0
            certified[:, i] = h0[i]
            continue
        S = np.linalg.norm(geometry.J0 @ u[indices])
        bounds[:, i] = [speed_capped_bound(t, S, result.envelopes.C, result.envelopes.V)
                        for t in times]
        certified[:, i] = h0[i] - bounds[:, i]
    global_lower = certified.min(axis=1)
    first_collision = next((float(times[k]) for k in np.flatnonzero(actual_min <= 0)), None)
    moving = [i for i, row in enumerate(rows) if row[2] is not None]
    residual = float(np.max((displacement-bounds)[:, moving]))
    safe_prefix = float(actual_min[times <= tau+1e-12].min())

    fig = plt.figure(figsize=(15.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1, 1])
    ax = fig.add_subplot(gs[0], projection="3d")
    ghost_times = np.linspace(0, tau, 5)
    colors = plt.cm.Blues(np.linspace(.28, .90, len(ghost_times)))
    paths = []
    for gi, t in enumerate(ghost_times):
        transforms, frames, cc = robot.fk(q0+t*u)
        joints = np.array([frames[i][0] for i in range(6)])
        tool = cc[target]
        chain = np.vstack((joints, tool)); paths.append(tool)
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], "-o", color=colors[gi],
                alpha=.25+.75*gi/(len(ghost_times)-1), lw=2.8, ms=4)
        ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], s=(radii*520)**2,
                   color=colors[gi], alpha=.035+.11*gi/(len(ghost_times)-1), edgecolors="none")
    paths = np.array(paths)
    ax.plot(paths[:, 0], paths[:, 1], paths[:, 2], color="#1565c0", lw=3)
    ox, oy, oz = sphere_mesh(obstacle, obstacle_radius)
    ax.plot_surface(ox, oy, oz, color="#d1495b", alpha=.58, linewidth=0)
    ax.scatter(*c0[binding], s=105, color="#f0a202", edgecolor="black", zorder=8)
    ax.text(*c0[binding], "  binding sphere", fontsize=9)
    ax.set_title(f"Exact UR5 motion, 0 to certified $\\tau={tau:.3f}$ s")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    allp = np.vstack((centers[times <= max(tau, .15), binding], obstacle[None, :],
                      np.array([frames0[i][0] for i in range(6)])))
    mid = (allp.max(0)+allp.min(0))/2; span = max(np.ptp(allp, axis=0))*.62
    ax.set_xlim(mid[0]-span, mid[0]+span); ax.set_ylim(mid[1]-span, mid[1]+span)
    ax.set_zlim(max(0, mid[2]-span), mid[2]+span); ax.view_init(23, -53)

    ax = fig.add_subplot(gs[1])
    ax.plot(times, actual_min, color="#172a3a", lw=2.5, label="true minimum clearance")
    ax.plot(times, global_lower, color="#e07a00", lw=2.5, label="certified lower bound")
    ax.axhline(0, color="black", lw=.8); ax.axvspan(0, tau, color="#4caf50", alpha=.10)
    ax.axvline(tau, color="#2e7d32", ls="--", lw=1.7, label=f"certified hold $\\tau$")
    if first_collision is not None:
        ax.axvline(first_collision, color="#b3261e", ls=":", lw=1.7,
                   label=f"first sampled collision ({first_collision:.3f} s)")
    ax.set(xlabel="time along $q(t)=q_0+tu$ (s)", ylabel="clearance (m)",
           title="The lower bound stays below true clearance")
    ax.grid(alpha=.22); ax.legend(fontsize=8, loc="lower left")

    ax = fig.add_subplot(gs[2])
    i = binding
    ax.plot(times, displacement[:, i], color="#1565c0", lw=2.5,
            label="true sphere displacement")
    ax.plot(times, bounds[:, i], color="#e07a00", lw=2.5,
            label="$D(t)$, computed envelope")
    ax.axhline(h0[i], color="#b3261e", ls=":", lw=2,
               label=f"initial clearance $h_0={h0[i]:.3f}$ m")
    ax.axvline(tau, color="#2e7d32", ls="--", lw=1.7)
    ax.scatter([tau], [h0[i]], s=55, color="#2e7d32", zorder=5)
    ax.set(xlabel="time (s)", ylabel="distance (m)",
           title=r"$D(t)\geq\|p(t)-p(0)\|$ and $D(\tau)=h_0$")
    ax.grid(alpha=.22); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("UR5 kinematic-envelope hold certificate — computed from URDF geometry",
                 fontsize=15, weight="bold")
    fig.savefig(OUT/"ur5_kinematic_envelope.png", dpi=220)
    fig.savefig(OUT/"ur5_kinematic_envelope.pdf")

    audit = {
        "urdf": str(URDF), "q0_rad": q0.tolist(), "u_rad_per_s": u.tolist(),
        "horizon_s": horizon, "derived_obstacle_center_m": obstacle.tolist(),
        "obstacle_radius_m": obstacle_radius, "target_sphere": target,
        "binding_sphere": binding, "binding_link": robot.spheres[binding][0],
        "binding_radius_m": radii[binding], "binding_initial_clearance_m": h0[binding],
        "certified_hold_s": tau, "first_sampled_collision_s": first_collision,
        "minimum_true_clearance_on_certified_prefix_m": safe_prefix,
        "max_true_displacement_minus_envelope_m": residual,
        "samples": len(times),
        "statement": "For every sphere i, h_i(t) >= h_i(0)-D_i(t); tau is the minimum root across all world rows."
    }
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
