# Reachy2 right-arm navigation experiments (PyBullet)

The counterpart of `ur5_experiments/`, for Reachy2's **right arm (7 DoF)** in six
obstacle scenes, set up for **navigation in configuration space**: getting
`r_arm_tip` to goals that require routing the arm *around* geometry. It carries
the kinematics/collision layer, the scenes, an RRT-Connect planner, and a
`LiveSdfEnv` that swaps the analytic geometry for the **GP-SDF built from
Gazebo's head ToF** so the same experiments run against sensed geometry.

## Setup

This repo and [`reachy_sbc_experiment`][sbc] must be **siblings under one
parent directory** — every script resolves the spherized URDF, the SRDF and the
Cap'n Proto schema relative to `../../reachy_sbc_experiment`:

```
<parent>/
  ompl/                     <- this repo (branch `reachy-experiments`)
  reachy_sbc_experiment/    <- branch `gazebo-sdf-clutter-experiments`
```

`$REACHY_SBC_ROOT` overrides that for a checkout laid out differently.

```bash
git clone -b reachy-experiments        git@github.com:mamani828/ompl.git
git clone -b gazebo-sdf-clutter-experiments --recurse-submodules \
    git@github.com:lkm1321/reachy_sbc_experiment.git

python3 -m venv .venv && . .venv/bin/activate
pip install numpy pybullet pillow pycapnp     # pycapnp only for the live-SDF path
```

`pybullet` and `numpy` cover everything analytic. `pycapnp` is needed only by
`reachy_nav/live_sdf.py`, and `ffmpeg` only by `--video`.

[sbc]: https://github.com/lkm1321/reachy_sbc_experiment

## Quick start

```bash
python scripts/run_sim.py --env shelf              # interactive GUI
python scripts/run_sim.py --env all --headless     # all six scenes, no window
python scripts/reach_envelope.py                   # re-measure the workspace

python scripts/demo.py --env shelf --hold          # watch one scene
python scripts/demo.py --env all --video out/reachy_demo.mp4    # record an mp4
```

## Watching it: `scripts/demo.py`

Green spheres are goals. For each one the arm replays the **straight
joint-space line** from home -- the naive plan -- and the marker turns red on
the first step where a collision-model sphere makes contact. The red markers
are exactly the motions a planner has to replace.

`--video` renders offscreen with `getCameraImage` and encodes with ffmpeg
rather than grabbing the screen, so frames are deterministic, independent of
window size, and it works headless. `--png` saves a single frame per scene,
`--speed` scales the live playback, `--hold` keeps the window open.

Needs `pybullet` and `numpy`. The robot comes from the **spherized** URDF
(`--urdf`, default `reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/`),
the same artifact `src/ompl/robots/Reachy2.h` is generated from, so the Python
harness and the C++ planner share one geometry.

## What the run reports

```
=== shelf ===
Two-bay shelf; goals require entering horizontal slots.  (6 obstacle bodies, 4 goals,
24/85 movable spheres, 740 self pairs checked)
  goal 1 [ 0.53 -0.22 0.95] (lower bay, inboard): reachable (clearance +0.0284 m
    [world: r_hand_palm_link], IK err 1.2 mm) | straight line COLLISION at step 44/65,
    min clearance -0.0600 m
  -> 4/4 goal configs reachable & collision-free; 4/4 straight-line paths hit something
```

## Current numbers (seed 0)

| scene | goals | reachable & collision-free | straight line collides |
| --- | --- | --- | --- |
| `empty` | 4 | 4/4 | 0/4 |
| `shelf` | 4 | 4/4 | **4/4** |
| `wall_gap` | 2 | 2/2 | **2/2** |
| `corridor` | 2 | 2/2 | **2/2** |
| `pillars` | 3 | 3/3 | **3/3** |
| `clutter` | 3 | 3/3 | **3/3** |
| **total** | **18** | **18/18** | **14/18** |

`empty` scoring 0/4 is the point of `empty`: with nothing in the way a straight
joint-space line is a valid plan, so it is the control. Every other scene defeats
it on every goal.

Two separate numbers, and the gap between them is the point:

- **reachable** — a collision-free configuration exists at the goal pose.
- **straight-line collisions** — how many goals a naive straight line in joint
  space fails to reach without hitting something.

That second number is the planner-shaped hole.

## Why these scenes are re-sited rather than rescaled

Reachy is not a UR5 on a table, and three measurements force the geometry to
move. From the `r_arm_tip` envelope over 30000 uniform in-limits samples
(`scripts/reach_envelope.py`), in the world frame the C++ side uses:

```
x: -0.661 .. +0.646   p5 -0.453  p50 +0.046  p95 +0.492
y: -0.858 .. +0.247   p5 -0.808  p50 -0.479  p95 -0.005
z: +0.461 .. +1.763   p5 +0.688  p50 +1.195  p95 +1.617
```

- **Forward reach falls off a cliff.** `x > 0.45` is 7.3% of samples, `x > 0.60`
  is 0.9%, and `x > 0.65` never occurs. The UR5 scenes place obstacles out to
  x = 0.7 on a 0.9144 m table; put there for Reachy they are pure scenery. Note
  the existing C++ demo's shelf at x = 0.62 sits at the very edge of this.
- **The workspace is one-sided.** The right arm is centred at y = −0.48 and
  crosses the midline only to y = +0.247, so a y-symmetric scene wastes half its
  geometry. Every scene here is biased to −y.
- **There is no table.** Reachy stands on the floor with `base_link` at
  z = 0.1075 m, so obstacles are floor-standing and goal heights come from the
  arm's own z band (≈ 0.85–1.45) rather than from a table top.

## Design notes

Five things were non-obvious enough to be worth stating, because each one
silently produced wrong results first.

**The TCP is `r_arm_tip`.** Not `r_hand_palm_link`, not `r_arm_tip_bottom`. It is
the frame `reachy2_symbolic_ik` solves for and the one the C++ demo reports. This
is the Reachy equivalent of the UR5 flange/TCP trap, and it fails the same way:
silently, by centimetres.

**`p.calculateJacobian` did not agree with the kinematics.** Its columns came back
rotated −90° about x relative to a numerical differentiation of `getLinkState`,
so DLS steps drove the tip sideways and IK solved **2/20** self-consistency cases
at 235 mm median error. `ReachyArm.jacobian` uses finite differences instead:
20/20 at 0.37 mm. Seven extra FK calls per iteration are cheap next to that.

**IK needs a seeded pool, not random restarts.** A DLS solve from an arbitrary
seed wanders; started from a sampled configuration whose tip is already
centimetres away it converges in a few iterations. `ReachyArm.sample_pool`
precomputes 3000 (config, tip) pairs and goals are seeded nearest-tip-first —
the same trick the C++ mobile demo uses.

**Mask to the spheres the arm can move.** Only **24 of 85** spheres move with
`r_arm`; the rest — wheels, lidar, support bars, the other arm, the head — are
rigid for this group. Including them does not merely waste work: a rigid pair
puts a floor under the reported minimum, which is why an obstacle-free scene
reported an identical `+0.0155 m` at every goal.

**A benchmark scene has to be checked, not just written.** Three of the six were
wrong on the first run and each failed differently. `shelf` at the C++ demo's
x = 0.62 lost its two outboard goals to IK by 11-26 mm, because that x is the
edge of the envelope — it moved to 0.53. `pillars` put a post 1.3 mm inside a
goal configuration, because the rejection test compared axis-aligned centre
distances instead of a keep-out radius around the goal and its approach column.
And `clutter` was measuring nothing at all: purely random boxes left **0/3**
goals blocked, so it now seeds obstacles on the workspace segment from the home
tip (measured at `[0.0197, -0.3682, 0.4660]`) to each goal before filling in at
random. After the fixes: 18/18 reachable, 14/18 straight-line collisions.

**Use the SRDF for self-collision pairs.** Excluding only same-link pairs is not
enough. It leaves adjacent links across a joint (`r_elbow_arm_link` vs
`r_elbow_forearm_link`, which reported a *negative* clearance at perfectly good
configurations) and the mimic-coupled gripper (`r_hand_palm_link` vs
`r_hand_proximal_link`, a constant, because those fingers never move). Reading
`reachy2.srdf` cut the table from 1659 pairs to 740 and turned `empty` from
2/4 into the 4/4 it should always have been. Pairs are then further filtered to
those with ≥ 20 mm of headroom at the reference pose, as the C++ demo does: a
barrier cannot establish an invariant that is already negative at the start.

## Layout

| File | Role |
| --- | --- |
| `reachy_nav/robot.py` | URDF staging, limits, FK, sphere placement, FD Jacobian, DLS IK, sample pool |
| `reachy_nav/scene.py` | `Box` primitives with exact signed distance, `Env`/`Goal` |
| `reachy_nav/envs.py` | the six scenes, sited from the measured envelope |
| `reachy_nav/motion.py` | movable-sphere mask, SRDF pair table, clearance, straight-line baseline |
| `reachy_nav/planner.py` | RRT-Connect with preallocated trees, shortcutting, densify |
| `reachy_nav/live_sdf.py` | `LiveSdfEnv` over the Cap'n Proto GP-SDF + `GridCache` |
| `scripts/run_sim.py` | the harness and its two headline numbers |
| `scripts/reach_envelope.py` | re-measure the reachable workspace |
| `scripts/demo.py` | replay/record; `--plan` uses the planner instead of a straight line |
| `scripts/pick_demo.py`, `scripts/bimanual_pick_demo.py` | shelf picking, one arm and two |
| `scripts/cache_sdf.py` | sample the live GP-SDF onto a grid, save `.npz` |
| `scripts/planning_experiment.py` | batch planning against a cached map, scored on truth |
| `scripts/gazebo_sdf_experiment.py` | plan against the live map and **execute in Gazebo** |
| `scripts/clutter_experiment.py` | the same for `clutter.world` |
| `scripts/replay_video.py` | re-render a saved trajectory |

## Planning against the live GP-SDF

`LiveSdfEnv` is a drop-in `Env`: same `.distance(pts)` / `.clearance(centers,
radii)`, answered by the running `sdf_mapping` container over Cap'n Proto RPC on
port 51111 instead of by analytic primitives. `GridCache` samples it once onto a
grid (60x68x55 at 0.03 m) and interpolates, which is what makes batch
experiments practical — a cached map costs 0.9 MB and is checked in under
`out/`.

```bash
# stack up first, in reachy_sbc_experiment:
REACHY_WORLD=clutter.world docker compose --env-file config/scenarios/sim.env up -d

python scripts/cache_sdf.py --out out/clutter_map.npz          # ~4 min of mapping
python scripts/planning_experiment.py --map out/clutter_map.npz
python scripts/clutter_experiment.py                           # plan + execute in Gazebo
```

`clutter.world` ships in `reachy_sbc_experiment` under `worlds/` and has to be
copied into the submodule's world directory first — that repo's README says how.

### What this measured, and it is the headline result

Planning against the **sensed** map is not equivalent to planning against the
geometry, and the planner is not the reason:

| | analytic geometry | cached GP-SDF |
| --- | --- | --- |
| same 3 goals, same seed | **3/3 collision-free** | **3/3 collide in truth** |

Scored over larger batches: `shelf` 20 goals -> 20% refused, **69% of returned
paths collide against true geometry**, worst penetration -0.0386 m; `clutter` 30
goals -> 53% refused, **93% collide**, worst -0.093 m. Every world collision was
a hand sphere.

The cause is **obstacle interiors read as free space**. Probing points strictly
inside each body:

| body | true distance | map says | interior reading free |
| --- | --- | --- | --- |
| `post_left` (0.16 m) | -0.048 | **+0.195** | 97.5% |
| `post_mid` (0.16 m) | -0.047 | +0.112 | 82.5% |
| `block_low` (0.10 m) | -0.027 | -0.001 | 47.5% |
| `target_object` (0.07 m) | -0.020 | -0.003 | 20% |

Smaller bodies map better because the map's ~80 mm surface shell covers
proportionally more of them; thickening the posts from 0.06 to 0.16 m made
penetrations *worse* (-0.084 -> -0.117), which is the signature of an unmapped
core rather than of noise.

Three things were ruled out, each with its own measurement: the grid cache
(2.85 mm median against live RPC), the frame conversion (torso at 1.1035 in the
harness vs 0.996 in Gazebo, differing by exactly `BASE_Z` = 0.1075), and the
planner itself (the table above).

`queryOcc` is not a fix — interior coverage by `sdf<0` / `occ>0.5` / either is
3.5/6.0/7.0% for `post_left`. Nor is "only trust the front": the occupancy
log-odds reads **-100 everywhere**, including occluded volume and space behind
the robot, so the map has no *unknown* state to reason from — unobserved volume
is asserted free.

**This is structural, not a tuning problem.** A ToF only ever returns front
surfaces, so nothing observes an obstacle's interior on hardware either. Per
Brian: tune the SDF on hardware, not against the simulator — the mechanism
transfers, the magnitudes above do not.

## Adding a planner

`Env.distance(points)` is an exact analytic signed distance field and
`Env.clearance(centers, radii)` is the per-sphere version the barrier consumes —
evaluated from the primitives, never sampled from collision queries, for the
reason `ur5_experiments/scripts/export_scene.py` gives: the CBF consumes the SDF
*gradient*, and only an analytic field gets that right. A planner needs
`arm.lo`/`arm.hi`, `worst_contact`, and a start and goal configuration.

## Not done yet

- **No `.grid`/`.problem` export.** `ur5_experiments/scripts/export_scene.py`
  bakes scenes for `demos/UR5PyBulletSceneDemo.cpp`; the Reachy equivalent and a
  `Reachy2PyBulletSceneDemo.cpp` do not exist, so these scenes are not yet
  reachable from the C++ CBF planner.
- **No smoke test.** `ur5_experiments/scripts/smoke_test.py` checks the
  randomised scenes stay solvable across seeds; **only seed 0 has been run
  here**, so the keep-out guarantee for `pillars` and `clutter` is asserted by
  construction rather than verified. That is the first thing to add.
- **No geometric ray-carving.** The only proposal with a chance of moving the
  93% collision rate: mark the volume behind the first mapped surface, and
  outside the camera frustum, as occupied rather than free. Diagnosed, designed,
  not built.
- **Nothing has been run on hardware.** The harness is sim-independent — point
  `LiveSdfEnv` at `r2-0008`'s `sdf_mapping` by changing the host — but scoring
  "did the path collide" needs ground-truth geometry a real scene does not hand
  you. On hardware the measurable proxies are refusal rate, plan/execute success
  and map self-consistency.
