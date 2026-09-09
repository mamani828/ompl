#!/usr/bin/env python3
"""Plot exact and certified displacement regions for a two-link example."""

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

# Obstacles: one row per obstacle center.
OBS = np.array([
    [1.15, 0.55],
    [0.35, 0.45],
])

# Matching obstacle radii.
RADII = np.array([
    0.23,
    0.18,
])


def endpoints(q):
    a = np.cumsum(q)
    links = L[:, None] * np.column_stack((np.cos(a), np.sin(a)))
    return np.vstack((np.zeros(2), np.cumsum(links, axis=0)))


def clearance(dq):
    """Minimum end-effector clearance over all obstacles."""
    ee = endpoints(Q0 + dq)[-1]
    clearances = np.linalg.norm(OBS - ee, axis=1) - RADII
    return float(np.min(clearances))


def model():
    p = endpoints(Q0)
    s = np.column_stack((np.diff(p, axis=0), np.zeros(2)))
    return Geometry(
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        s,
    ), p


def first_exit(u, limit, samples):
    """Numerically find first collision along displacement ray u."""
    previous = 0.0

    for t in np.linspace(limit / samples, limit, samples):
        if clearance(t * u) <= 0:
            low, high = previous, t

            for _ in range(45):
                mid = (low + high) / 2

                if clearance(mid * u) > 0:
                    low = mid
                else:
                    high = mid

            return low

        previous = t

    return limit


def boundaries(limit, rays, samples):
    g, p = model()

    angles = np.linspace(0, 2 * math.pi, rays, endpoint=False)
    directions = np.column_stack((
        np.cos(angles),
        np.sin(angles),
    ))

    l1 = np.empty(rays)
    l2 = np.empty(rays)
    combined = np.empty(rays)
    truth = np.empty(rays)

    # Initial vector from end effector to each obstacle center.
    separations = np.column_stack((
        OBS[:, 0] - p[-1, 0],
        OBS[:, 1] - p[-1, 1],
        np.zeros(len(OBS)),
    ))

    distances = np.linalg.norm(separations, axis=1)
    normals = separations / distances[:, None]
    d0s = distances - RADII

    if np.any(d0s <= 0):
        raise RuntimeError(
            "Initial configuration is already in collision with an obstacle: "
            f"d0={d0s}"
        )

    for i, u in enumerate(directions):
        obstacle_l1 = []
        obstacle_l2 = []
        obstacle_combined = []

        for d0, normal in zip(d0s, normals):
            t_l1 = min(
                limit,
                weighted_l1(d0, u, g.reach_radii),
            )

            t_l2 = level2(
                g,
                u,
                d0,
                limit,
            ).tau

            projected = normal_anchored_time(
                g,
                u,
                d0,
                limit,
                normal,
            )

            obstacle_l1.append(t_l1)
            obstacle_l2.append(t_l2)

            # Best certificate for this particular obstacle.
            obstacle_combined.append(
                max(t_l1, t_l2, projected)
            )

        # To remain collision-free, the certificate must hold
        # simultaneously for every obstacle.
        l1[i] = min(obstacle_l1)
        l2[i] = min(obstacle_l2)
        combined[i] = min(obstacle_combined)

        truth[i] = first_exit(
            u,
            limit,
            samples,
        )

    return (
        directions,
        l1,
        l2,
        combined,
        truth,
        d0s,
        p,
    )


def polygon(u, radius):
    p = u * radius[:, None]
    return np.vstack((p, p[0]))


def plot(output, limit, resolution, rays, samples):
    (
        u,
        l1,
        l2,
        combined,
        truth,
        d0s,
        joints,
    ) = boundaries(
        limit,
        rays,
        samples,
    )

    # ------------------------------------------------------------------
    # Exact configuration-space collision map
    # ------------------------------------------------------------------

    axis = np.linspace(-limit, limit, resolution)
    x, y = np.meshgrid(axis, axis)

    q1 = Q0[0] + x
    q2 = Q0[1] + y

    px = (
        L[0] * np.cos(q1)
        + L[1] * np.cos(q1 + q2)
    )

    py = (
        L[0] * np.sin(q1)
        + L[1] * np.sin(q1 + q2)
    )

    # Safe only if the end effector avoids EVERY obstacle.
    safe = np.ones_like(px, dtype=bool)

    for obs, radius in zip(OBS, RADII):
        obstacle_safe = (
            (px - obs[0]) ** 2
            + (py - obs[1]) ** 2
            >= radius ** 2
        )

        safe &= obstacle_safe

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, (ws, region) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.7),
        gridspec_kw={
            "width_ratios": [0.82, 1.35],
        },
    )

    # ------------------------------------------------------------------
    # Workspace panel
    # ------------------------------------------------------------------

    ws.plot(
        joints[:, 0],
        joints[:, 1],
        "-o",
        color="#171717",
        lw=3,
        ms=5,
    )

    # End-effector marker.
    ws.add_patch(
        Circle(
            joints[-1],
            0.10,
            color="#4c6edb",
            alpha=0.4,
        )
    )

    # Draw every obstacle and its initial separation line.
    for obs, radius in zip(OBS, RADII):
        ws.add_patch(
            Circle(
                obs,
                radius,
                color="#df5b35",
                alpha=0.8,
            )
        )

        ws.plot(
            [joints[-1, 0], obs[0]],
            [joints[-1, 1], obs[1]],
            "--",
            color="#d17b23",
            alpha=0.7,
        )

    d0_text = ", ".join(
        f"{d:.3f}" for d in d0s
    )

    ws.set(
        title=rf"workspace, $d_0=[{d0_text}]$",
        xlabel="$x$",
        ylabel="$y$",
        xlim=(-0.15, 1.65),
        ylim=(-0.35, 1.15),
    )

    ws.set_aspect("equal")
    ws.grid(alpha=0.18)

    # ------------------------------------------------------------------
    # Configuration-space panel
    # ------------------------------------------------------------------

    region.contourf(
        x,
        y,
        safe.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=[
            "#e9c3cc",
            "#c8eee2",
        ],
        alpha=0.72,
    )

    # Exact ray-wise first-exit region.
    region.fill(
        *polygon(u, truth).T,
        color="#65c9ad",
        alpha=0.16,
    )

    # Certified combined region.
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
        poly = polygon(u, radii)

        region.plot(
            poly[:, 0],
            poly[:, 1],
            color=color,
            lw=width,
        )

    # Nominal configuration.
    region.plot(
        0,
        0,
        "o",
        color="#151515",
        ms=4,
    )

    region.set(
        title="exact and certified displacement regions",
        xlabel=r"$\delta q_1$ [rad]",
        ylabel=r"$\delta q_2$ [rad]",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
    )

    region.set_aspect("equal")
    region.grid(alpha=0.16)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    fig.legend(
        handles=[
            Patch(
                color="#c8eee2",
                label="exact safe configurations",
            ),
            Patch(
                color="#e9c3cc",
                label="exact collision",
            ),
            Line2D(
                [],
                [],
                color="#258a72",
                lw=2,
                label="true first exit per ray",
            ),
            Line2D(
                [],
                [],
                color="#6650b5",
                lw=2,
                label="max(L1, Level-2, normal anchored), then min over obstacles",
            ),
            Line2D(
                [],
                [],
                color="#3975c6",
                lw=1.5,
                label="Level-2",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.5,
                label="old weighted L1",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(
        rect=(0, 0.13, 1, 1)
    )

    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=190,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"wrote {output}")

    # ------------------------------------------------------------------
    # Validation / metrics
    # ------------------------------------------------------------------

    exiting = truth < limit - 1e-12

    if np.any(combined > truth + 1e-10):
        bad = np.flatnonzero(
            combined > truth + 1e-10
        )

        worst = bad[
            np.argmax(
                combined[bad] - truth[bad]
            )
        ]

        raise RuntimeError(
            "sampled certified boundary exceeds exact first exit; "
            f"ray={worst}, "
            f"certified={combined[worst]:.12f}, "
            f"truth={truth[worst]:.12f}"
        )

    if np.any(exiting):
        print(
            f"d0={np.array2string(d0s, precision=6)}; "
            f"{np.count_nonzero(exiting)} exiting rays; "
            f"median/true: "
            f"L1={np.median(l1[exiting] / truth[exiting]):.3f}, "
            f"Level-2={np.median(l2[exiting] / truth[exiting]):.3f}, "
            f"combined={np.median(combined[exiting] / truth[exiting]):.3f}"
        )
    else:
        print(
            f"d0={np.array2string(d0s, precision=6)}; "
            "no rays exited within the requested radius"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

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
