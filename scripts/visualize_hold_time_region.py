#!/usr/bin/env python3
"""Plot exact and certified displacement regions for a sphere-model 2R arm.

Certificate semantics:
  - world/environment rows: max(weighted L1, Level-2)
  - self-collision rows:    max(weighted L1, Level-2, normal-anchored)
  - global certificate:     min over all collision rows
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hold_time import Geometry, level2, normal_anchored_time, weighted_l1


L = np.array([0.90, 0.65])
Q0 = np.array([0.45, -1.00])

OBS = np.array([
    [1.15, 0.55],
    [0.55, 0.75],   # moved upward so Q0 is collision-free
])
RADII = np.array([0.13, 0.18])

SPHERES_PER_LINK = 5
SPHERE_OVERLAP = 0.15
ELBOW_IGNORE_STEPS = 1


def tight_sphere_chain(
    length,
    n=SPHERES_PER_LINK,
    overlap=SPHERE_OVERLAP,
):
    """Sphere radius + normalized centers for a tight overlapping chain."""
    if n < 1:
        raise ValueError("SPHERES_PER_LINK must be >= 1")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("SPHERE_OVERLAP must satisfy 0 <= overlap < 1")

    if n == 1:
        return 0.5 * length, np.array([0.5])

    radius = length / (2.0 + 2.0 * (n - 1) * (1.0 - overlap))
    spacing = 2.0 * radius * (1.0 - overlap)
    centers = radius + np.arange(n) * spacing
    return radius, centers / length


LINK1_RADIUS, SPHERE_T1 = tight_sphere_chain(L[0])
LINK2_RADIUS, SPHERE_T2 = tight_sphere_chain(L[1])


def vec3(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[0], v[1], 0.0], dtype=float)


def endpoints(q):
    a = np.cumsum(q)
    links = L[:, None] * np.column_stack((np.cos(a), np.sin(a)))
    return np.vstack((np.zeros(2), np.cumsum(links, axis=0)))


def robot_spheres(q):
    """Return sphere centers on link 1 and link 2."""
    q1, q2 = q
    e1 = np.array([math.cos(q1), math.sin(q1)])
    e2 = np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
    elbow = L[0] * e1

    link1 = SPHERE_T1[:, None] * L[0] * e1
    link2 = elbow + SPHERE_T2[:, None] * L[1] * e2
    return link1, link2


def ignore_self_pair(i, j):
    """
    Ignore neighboring cross-link sphere pairs around the shared elbow.

    With ELBOW_IGNORE_STEPS=1 this ignores:
      link1[-1] vs link2[0]
      link1[-2] vs link2[0]
      link1[-1] vs link2[1]
    """
    from_elbow_link1 = (SPHERES_PER_LINK - 1) - i
    from_elbow_link2 = j
    return from_elbow_link1 + from_elbow_link2 <= ELBOW_IGNORE_STEPS


def clearance_components(q):
    """Return minimum world clearance and minimum self clearance."""
    link1, link2 = robot_spheres(q)

    world_min = np.inf
    for centers, robot_radius in (
        (link1, LINK1_RADIUS),
        (link2, LINK2_RADIUS),
    ):
        clearances = (
            np.linalg.norm(
                centers[:, None, :] - OBS[None, :, :],
                axis=2,
            )
            - robot_radius
            - RADII[None, :]
        )
        world_min = min(world_min, float(np.min(clearances)))

    self_min = np.inf
    for i, a in enumerate(link1):
        for j, b in enumerate(link2):
            if ignore_self_pair(i, j):
                continue
            d = np.linalg.norm(a - b) - LINK1_RADIUS - LINK2_RADIUS
            self_min = min(self_min, float(d))

    return float(world_min), float(self_min)


def clearance(dq):
    world, self_col = clearance_components(Q0 + dq)
    return min(world, self_col)


def first_exit(u, limit, samples):
    """Numerically find first world or self collision on q(t)=Q0+t*u."""
    previous = 0.0

    for t in np.linspace(limit / samples, limit, samples):
        if clearance(t * u) <= 0.0:
            low, high = previous, t
            for _ in range(45):
                mid = 0.5 * (low + high)
                if clearance(mid * u) > 0.0:
                    low = mid
                else:
                    high = mid
            return low
        previous = t

    return limit


def make_geometry(segments):
    segments = np.asarray(segments, dtype=float)
    if segments.ndim == 1:
        segments = segments[None, :]
    axes = np.tile([0.0, 0.0, 1.0], (len(segments), 1))
    return Geometry(axes, segments)


def certificate_times(
    geometry,
    velocity,
    d0,
    limit,
    kind,
    normal=None,
):
    """
    Return weighted-L1, Level-2, and the per-row combined certificate.

    World/environment rows:
        max(L1, Level-2)

    Self-collision rows:
        max(L1, Level-2, normal-anchored)

    The normal-anchored certificate is deliberately not applied to the
    environment. It is only used for self-collision rows, where the pairwise
    collision geometry is assumed convex.
    """
    velocity = np.asarray(velocity, dtype=float)

    if kind not in {"world", "self"}:
        raise ValueError(f"unknown collision-row kind: {kind!r}")

    if d0 <= 0.0:
        return 0.0, 0.0, 0.0

    if np.all(np.abs(velocity) <= 1e-15):
        return limit, limit, limit

    t_l1 = min(
        limit,
        float(weighted_l1(d0, velocity, geometry.reach_radii)),
    )

    t_l2 = min(
        limit,
        float(level2(geometry, velocity, d0, limit).tau),
    )

    if kind == "self":
        if normal is None:
            raise ValueError(
                "self-collision rows require a normal for "
                "normal_anchored_time()"
            )

        t_normal = min(
            limit,
            float(
                normal_anchored_time(
                    geometry,
                    velocity,
                    d0,
                    limit,
                    normal,
                )
            ),
        )

        t_best = max(t_l1, t_l2, t_normal)
    else:
        # No normal-anchored environment/world certificate.
        t_best = max(t_l1, t_l2)

    return t_l1, t_l2, t_best


def collision_constraints():
    """
    Build all collision-pair rows at Q0.

    World:
      link1 sphere -> q1
      link2 sphere -> q1,q2

      These rows use weighted L1 + Level-2 only.
      We intentionally do not construct/store environment normals.

    Self:
      link1 sphere vs link2 sphere -> q2-only relative sub-chain.
      q1 rotates both spheres rigidly and cancels from relative separation.

      These rows additionally use the normal-anchored certificate.
    """
    joints = endpoints(Q0)
    base, elbow = joints[0], joints[1]
    link1, link2 = robot_spheres(Q0)
    constraints = []

    # Link 1 spheres vs obstacles.
    for i, center in enumerate(link1):
        geometry = make_geometry([vec3(center - base)])

        for j, (obs, obs_radius) in enumerate(zip(OBS, RADII)):
            distance = np.linalg.norm(obs - center)
            d0 = distance - LINK1_RADIUS - obs_radius

            constraints.append({
                "label": f"world link1[{i}] / obs[{j}]",
                "kind": "world",
                "geometry": geometry,
                "active": np.array([0], dtype=int),
                "d0": float(d0),
            })

    # Link 2 spheres vs obstacles.
    for i, center in enumerate(link2):
        geometry = make_geometry([
            vec3(elbow - base),
            vec3(center - elbow),
        ])

        for j, (obs, obs_radius) in enumerate(zip(OBS, RADII)):
            distance = np.linalg.norm(obs - center)
            d0 = distance - LINK2_RADIUS - obs_radius

            constraints.append({
                "label": f"world link2[{i}] / obs[{j}]",
                "kind": "world",
                "geometry": geometry,
                "active": np.array([0, 1], dtype=int),
                "d0": float(d0),
            })

    # Link 1 vs link 2 self-collision.
    for i, a in enumerate(link1):
        for j, b in enumerate(link2):
            if ignore_self_pair(i, j):
                continue

            separation = a - b
            distance = np.linalg.norm(separation)
            d0 = distance - LINK1_RADIUS - LINK2_RADIUS

            # In link-1's frame, sphere a is fixed and b moves only about q2.
            geometry = make_geometry([vec3(b - elbow)])

            constraints.append({
                "label": f"self link1[{i}] / link2[{j}]",
                "kind": "self",
                "geometry": geometry,
                "active": np.array([1], dtype=int),
                "d0": float(d0),
                "normal": vec3(separation / distance),
            })

    return constraints, joints


def boundaries(limit, rays, samples):
    constraints, joints = collision_constraints()

    bad = [c for c in constraints if c["d0"] <= 0.0]
    if bad:
        details = "\n".join(
            f"  {c['label']}: d0={c['d0']:.9f}"
            for c in bad
        )
        raise RuntimeError(
            "Q0 is already in collision under the sphere model:\n" + details
        )

    angles = np.linspace(0.0, 2.0 * math.pi, rays, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    l1 = np.empty(rays)
    l2 = np.empty(rays)
    combined = np.empty(rays)
    truth = np.empty(rays)
    binding_labels = []

    for i, u in enumerate(directions):
        ray_l1 = limit
        ray_l2 = limit
        ray_combined = limit
        binding_label = None

        for c in constraints:
            velocity = u[c["active"]]

            t_l1, t_l2, t_best = certificate_times(
                c["geometry"],
                velocity,
                c["d0"],
                limit,
                c["kind"],
                c.get("normal"),
            )

            ray_l1 = min(ray_l1, t_l1)
            ray_l2 = min(ray_l2, t_l2)

            # Per row:
            #   world -> max(L1, Level-2)
            #   self  -> max(L1, Level-2, normal)
            #
            # Globally:
            #   min over all collision rows.
            if t_best < ray_combined:
                ray_combined = t_best
                binding_label = c["label"]

        l1[i] = ray_l1
        l2[i] = ray_l2
        combined[i] = ray_combined
        truth[i] = first_exit(u, limit, samples)
        binding_labels.append(binding_label)

    d0_world = min(
        c["d0"]
        for c in constraints
        if c["kind"] == "world"
    )
    d0_self = min(
        c["d0"]
        for c in constraints
        if c["kind"] == "self"
    )

    return (
        directions,
        l1,
        l2,
        combined,
        truth,
        d0_world,
        d0_self,
        joints,
        binding_labels,
    )


def polygon(u, radius):
    p = u * radius[:, None]
    return np.vstack((p, p[0]))


def exact_safe_grid(q1, q2):
    """Exact safe mask using the same sphere pairs as first_exit()."""
    safe = np.ones_like(q1, dtype=bool)

    e1x, e1y = np.cos(q1), np.sin(q1)
    e2x, e2y = np.cos(q1 + q2), np.sin(q1 + q2)
    elbow_x = L[0] * e1x
    elbow_y = L[0] * e1y

    link1 = [
        (
            t * L[0] * e1x,
            t * L[0] * e1y,
        )
        for t in SPHERE_T1
    ]

    link2 = [
        (
            elbow_x + t * L[1] * e2x,
            elbow_y + t * L[1] * e2y,
        )
        for t in SPHERE_T2
    ]

    # World collision.
    for cx, cy in link1:
        for obs, obs_radius in zip(OBS, RADII):
            r = LINK1_RADIUS + obs_radius
            safe &= (
                (cx - obs[0]) ** 2
                + (cy - obs[1]) ** 2
                >= r ** 2
            )

    for cx, cy in link2:
        for obs, obs_radius in zip(OBS, RADII):
            r = LINK2_RADIUS + obs_radius
            safe &= (
                (cx - obs[0]) ** 2
                + (cy - obs[1]) ** 2
                >= r ** 2
            )

    # Self collision.
    r_self = LINK1_RADIUS + LINK2_RADIUS
    for i, (ax, ay) in enumerate(link1):
        for j, (bx, by) in enumerate(link2):
            if ignore_self_pair(i, j):
                continue
            safe &= (
                (ax - bx) ** 2
                + (ay - by) ** 2
                >= r_self ** 2
            )

    return safe


def plot(output, limit, resolution, rays, samples):
    (
        u,
        l1,
        l2,
        combined,
        truth,
        d0_world,
        d0_self,
        joints,
        binding_labels,
    ) = boundaries(limit, rays, samples)

    axis = np.linspace(-limit, limit, resolution)
    x, y = np.meshgrid(axis, axis)
    q1 = Q0[0] + x
    q2 = Q0[1] + y
    safe = exact_safe_grid(q1, q2)

    fig, (ws, region) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.9),
        gridspec_kw={"width_ratios": [0.82, 1.35]},
    )

    # Workspace.
    ws.plot(
        joints[:, 0],
        joints[:, 1],
        "-o",
        color="#171717",
        lw=3,
        ms=5,
        zorder=5,
    )

    link1, link2 = robot_spheres(Q0)

    for center in link1:
        ws.add_patch(
            Circle(
                center,
                LINK1_RADIUS,
                facecolor="#7d8490",
                edgecolor="#333333",
                lw=0.7,
                alpha=0.55,
                zorder=3,
            )
        )

    for center in link2:
        ws.add_patch(
            Circle(
                center,
                LINK2_RADIUS,
                facecolor="#4c6edb",
                edgecolor="#333333",
                lw=0.7,
                alpha=0.55,
                zorder=3,
            )
        )

    all_centers = np.vstack((link1, link2))
    all_radii = np.r_[
        np.full(len(link1), LINK1_RADIUS),
        np.full(len(link2), LINK2_RADIUS),
    ]

    for obs, radius in zip(OBS, RADII):
        ws.add_patch(
            Circle(
                obs,
                radius,
                color="#df5b35",
                alpha=0.82,
                zorder=2,
            )
        )

        nominal_clearances = (
            np.linalg.norm(all_centers - obs, axis=1)
            - all_radii
            - radius
        )
        k = int(np.argmin(nominal_clearances))
        closest = all_centers[k]

        ws.plot(
            [closest[0], obs[0]],
            [closest[1], obs[1]],
            "--",
            color="#d17b23",
            alpha=0.65,
            lw=1.1,
            zorder=1,
        )

    ws.set(
        title=(
            "workspace sphere model\n"
            rf"$r_1={LINK1_RADIUS:.3f}$, "
            rf"$r_2={LINK2_RADIUS:.3f}$, "
            rf"$d_{{world}}={d0_world:.3f}$, "
            rf"$d_{{self}}={d0_self:.3f}$"
        ),
        xlabel="$x$",
        ylabel="$y$",
        xlim=(-0.15, 1.65),
        ylim=(-0.35, 1.15),
    )
    ws.set_aspect("equal")
    ws.grid(alpha=0.18)

    # Configuration-space region.
    region.contourf(
        x,
        y,
        safe.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=["#e9c3cc", "#c8eee2"],
        alpha=0.72,
    )

    region.fill(
        *polygon(u, truth).T,
        color="#65c9ad",
        alpha=0.16,
    )
    region.fill(
        *polygon(u, combined).T,
        color="#8b75d7",
        alpha=0.24,
    )

    for radii, color, width in [
        (truth, "#258a72", 2.0),
        (combined, "#6650b5", 1.8),
        (l2, "#3975c6", 1.5),
        (l1, "#555555", 1.5),
    ]:
        p = polygon(u, radii)
        region.plot(
            p[:, 0],
            p[:, 1],
            color=color,
            lw=width,
        )

    region.plot(
        0.0,
        0.0,
        "o",
        color="#151515",
        ms=4,
    )
    region.set(
        title="exact and certified sphere-collision regions",
        xlabel=r"$\delta q_1$ [rad]",
        ylabel=r"$\delta q_2$ [rad]",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
    )
    region.set_aspect("equal")
    region.grid(alpha=0.16)

    fig.legend(
        handles=[
            Patch(
                color="#c8eee2",
                label="exact safe: world + self",
            ),
            Patch(
                color="#e9c3cc",
                label="exact collision: world or self",
            ),
            Line2D(
                [],
                [],
                color="#258a72",
                lw=2,
                label="true first sphere contact per ray",
            ),
            Line2D(
                [],
                [],
                color="#6650b5",
                lw=2,
                label=(
                    "combined: world=max(L1,L2), "
                    "self=max(L1,L2,normal), then global min"
                ),
            ),
            Line2D(
                [],
                [],
                color="#3975c6",
                lw=1.5,
                label="global Level-2",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.5,
                label="global weighted L1",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"wrote {output}")

    # Diagnostics.
    initial_world, initial_self = clearance_components(Q0)
    print(
        "sphere model: "
        f"N/link={SPHERES_PER_LINK}, "
        f"overlap={SPHERE_OVERLAP:.0%}, "
        f"r1={LINK1_RADIUS:.6f}, "
        f"r2={LINK2_RADIUS:.6f}"
    )
    print(
        "initial minimum clearances: "
        f"world={initial_world:.6f}, "
        f"self={initial_self:.6f}"
    )

    exiting = truth < limit - 1e-12
    violations = combined > truth + 1e-10

    if np.any(violations):
        bad = np.flatnonzero(violations)
        worst = bad[
            np.argmax(combined[bad] - truth[bad])
        ]
        raise RuntimeError(
            "sampled certified boundary exceeds exact first exit; "
            f"ray={worst}, "
            f"u={u[worst]}, "
            f"certified={combined[worst]:.12f}, "
            f"truth={truth[worst]:.12f}, "
            f"binding={binding_labels[worst]!r}"
        )

    if np.any(exiting):
        print(
            f"{np.count_nonzero(exiting)} exiting rays; "
            f"median/true: "
            f"L1={np.median(l1[exiting] / truth[exiting]):.3f}, "
            f"Level-2={np.median(l2[exiting] / truth[exiting]):.3f}, "
            f"combined={np.median(combined[exiting] / truth[exiting]):.3f}"
        )
    else:
        print("no rays reached world or self collision within the radius")

    world_binds = sum(
        label is not None and label.startswith("world")
        for label in binding_labels
    )
    self_binds = sum(
        label is not None and label.startswith("self")
        for label in binding_labels
    )
    horizon_binds = rays - world_binds - self_binds

    print(
        "combined binding rows: "
        f"world={world_binds}, "
        f"self={self_binds}, "
        f"horizon={horizon_binds}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/hold_time_region.png",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.6,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=420,
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--exit-samples",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    plot(
        args.output,
        args.radius,
        args.resolution,
        args.rays,
        args.exit_samples,
    )


if __name__ == "__main__":
    main()

