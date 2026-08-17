# Reachy2 bimanual OMPL + CBF interface

## Mobile-manipulation demo

`demo_Reachy2MobileCBFPlanning` keeps the fixed-base benchmark below unchanged and
plans the coupled state `[base_x, base_y, base_yaw, 14 arm joints]`. Both rows use
geometric RRTConnect, the same deterministic 60/20/20 transit/goal/uniform sampler,
and the same precomputed pool of collision-free bimanual mobile IK states. The hand
goals are Cartesian, so the terminal base pose is free.

```bash
cmake --build . --target demo_Reachy2MobileCBFPlanning -j
./demos/demo_Reachy2MobileCBFPlanning \
  --seconds 10 --trials 3 --shortcut 0.1 \
  --output reachy2_mobile_cbf.path

# Workspace, start, and Cartesian goals are named options.
./demos/demo_Reachy2MobileCBFPlanning \
  --start-x -0.70 --start-y 0 --start-yaw 0 \
  --xmin -1.0 --xmax 0.35 --ymin -0.65 --ymax 0.65 \
  --left-goal 0.62 0.20 1.1344 \
  --right-goal 0.62 -0.20 1.1344
```

The output is rate-timed and synchronized: each segment lasts the maximum travel
time of any base or arm coordinate at 0.35 m/s translation, 0.6 rad/s yaw, and
1.2 rad/s arm speed. The viewer autodetects both this format and the legacy
fixed-base format:

```bash
python3 scripts/visualize_reachy2_cbf.py reachy2_mobile_cbf.path --gui --hold
python3 scripts/visualize_reachy2_cbf.py reachy2_mobile_cbf.path --png mobile.png
```

`demo_Reachy2CBFPlanning` plans the two Reachy2 arms together in a 14-dimensional
joint space. The left and right goals are Cartesian arm-tip positions. A
damped-least-squares IK stage converts them into a coupled joint-space goal,
then runs the same geometric OMPL `RRTConnect` two ways: ordinary straight-line
edges with a collision checker, and CBF-rollout edges with no collision checker.

The robot adapter in `src/ompl/robots/Reachy2.h` is generated from the
spherized URDF and SRDF. It contains:

- the two seven-joint arm chains and their URDF limits;
- exact analytic point Jacobians for all 85 collision spheres;
- the left and right `arm_tip_bottom` frames;
- the SRDF-filtered sphere-pair table for self-collision barriers.

All other Reachy joints are fixed at their URDF zero positions. This includes
the tripod, neck, antennas, and gripper joints.

## Build and run

```bash
cmake -S . -B build -DOMPL_BUILD_VAMP=OFF
cmake --build build --target demo_Reachy2CBFPlanning -j

# Built-in shelf with both default hand goals in the lower bay, 10 second limit.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path

# The third argument is rope-shortcut anchor spacing in radians for both rows.
# Zero disables shortcutting; 0.05 is the aggressive default.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_raw.path 0
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_short.path 0.05

# Custom left and right Cartesian targets.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path \
    0.62 0.20 1.1344  0.62 -0.20 1.1344
```

## Head-to-head wall time

Every invocation now runs both rows. The default is five trials and the table
reports median planner wall time. SDF baking, IK, path replay/audit, file output,
and optional shortcutting are deliberately outside the timed region. To change
the repeat count, append it after `shortcutRadians`:

```bash
# No shortcutting, nine timing trials.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path 0 9

# Custom targets, 0.05-rad CBF shortcut anchors, nine timing trials.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path \
    0.62 0.20 1.1344  0.62 -0.20 1.1344  0.05 9
```

Both rows use geometric `RRTConnect`, a 1.5-radian range, the same bounds and
goal tolerance, paired deterministic random seeds, and a matched 0.024-radian
sampling/audit resolution. Their execution order alternates between trials.
`ompl-rrtc` checks each straight-line edge with a value-only Reachy world/self
collision checker; `cbf-rrtc` constructs each edge as a filtered rollout and
uses an all-valid state checker. Evaluation counts therefore have different
units (collision checks versus filter calls); wall time is the directly
comparable number.

The Reachy filter follows the optimized UR5 path rather than the former
control-space implementation. Its matrices and qpmad workspace have fixed
capacity and are reused, rows are screened using configuration-independent
link-motion bounds, and qpmad is bypassed when no row can bind. Every call also
returns a conservative duration for which its control remains safe. The
filtered state space consumes that certificate to cover open parts of an edge
without another filter call, while dense replay still audits inside those
coarse spans. A per-trial table reports wall time, success, and planner-tree
nodes (`PlannerData` vertices) for both rows; the summary reports their medians.
It also reports QP calls, active rows, certified-step share, joint travel per
filter call, and replay misses so these optimizations remain observable.

On the default shelf target, one nine-seed run of the former fixed-step control
pipeline measured 6.50 ms for CBF versus 3.25 ms for ordinary OMPL, with the
ordinary row solving 8/9. The geometric certified-step pipeline measured
1.56 ms for CBF versus 1.71 ms for ordinary OMPL, with both solving 9/9. These
are machine-dependent timings, but the structural counters moved with them:
the representative CBF solve fell from roughly 315 filter calls to 65, about
43% of its remaining calls covered certified coarse spans, and replay reported
zero missing edges.

The CBF motion is written to the requested path. The ordinary OMPL motion is
written alongside it: `reachy2_cbf.path` produces `reachy2_ompl.path`; other
filenames receive an `.ompl` suffix. Both files use the same joint-name header
and can be passed to the viewer below.

The path file begins with the ordered joint names and then contains one
raw-radian 14-joint configuration per line. Because the requested goal is
Cartesian, an OMPL path that is approximate relative to one IK representative
is accepted only when both actual tips finish within 4 cm of their requested
positions. The executable returns nonzero when IK fails, that Cartesian test
fails, or the dense post-plan barrier audit fails.

Shortcutting is applied to both successful output paths with the same
0.05-radian anchor spacing. The ordinary row uses OMPL's stock geometric
`PathSimplifier::ropeShortcutPath`; its candidates are straight, densely
collision-checked edges. The CBF row uses the dimension-independent
`ompl::cbf::ropeShortcut` overload, whose candidates are complete Reachy CBF
rollouts and whose result retains every filtered integration waypoint. The CBF
pass accepts improvements down to 1% of the anchor spacing while preserving its
filtered-rollout and endpoint-arrival safety requirements. Both shortened paths
are densely audited before being written. The demo reports length before/after
and runtime for both, plus CBF rollout and arrival details.

## Visualize a path

The existing `ur5_experiments/scripts/replay_path.py` is specific to the UR5:
it assumes six UR5 joints, the UR5 TCP, and that experiment's shelf coordinates.
Use the Reachy2 replay adapter for this path format and scene:

```bash
# Interactive animation; leave the window open at the final pose.
python3 scripts/visualize_reachy2_cbf.py reachy2_cbf.path --gui --hold

# Headless final-frame render.
python3 scripts/visualize_reachy2_cbf.py reachy2_cbf.path \
    --png reachy2_cbf.png
```

The URDF supplied with Reachy2 points its visual meshes at absolute paths under
`/home/brian`. The viewer therefore shows the spherized robot by default, with
left/right/body spheres colored separately. If those meshes are installed and
the paths in a meshed URDF are valid, request them explicitly:

```bash
python3 scripts/visualize_reachy2_cbf.py reachy2_cbf.path --gui --hold \
    --meshed-urdf /path/to/reachy2_meshed.urdf
```

The shelf keeps the useful dimensions and world coordinates from
`ur5_experiments.ur5_nav.envs.env_shelf`: bottom shelf height 0.9144 m, shelf x
0.62 m, 0.14 m depth, 0.75 m width, and 0.44 m bay pitch. The UR5's solid
0.9144 m mounting table is deliberately omitted: Reachy is a floor-standing
mobile manipulator, not an arm bolted to that table. The shelf back and sides
extend to the floor instead.

Reachy's URDF defines `base_link` at the centre of the mobile base and places
the base visual 0.1075 m below it. Both the generated C++ model and the replay
tool therefore use a world z offset of 0.1075 m. `Reachy2` stores this as an
explicit base pose and also exposes `kinematicsAtBase`, so later mobile
manipulation can supply x/y/yaw from the RRT state without changing the arm's
generated geometry.

Paths produced before this base-frame correction use the old z=0 kinematics and
should be regenerated before replay; their joint values are not in the corrected
world frame.

The CBF row uses an all-valid OMPL state checker deliberately: environment and
self-collision safety are imposed by its filtered state-space interpolation and
matching motion validator. The ordinary OMPL row uses the same barrier geometry
as a value-only state validity checker.

## Regenerate from another Reachy2 description

```bash
python3 scripts/generate_reachy2_model.py \
    /path/to/reachy2_spherized.urdf \
    /path/to/reachy2.srdf \
    src/ompl/robots/Reachy2.h
```

The generator needs only Python's standard library. Rebuild the demo after
regeneration.

## Sphere-pair calibration

An SRDF allowed-collision matrix describes mesh collision semantics; it does
not guarantee that a fitted sphere model separates every remaining pair. A CBF
also cannot establish an invariant that is already negative at the declared
start configuration. The demo therefore retains semantic pairs that have at
least 20 mm of sphere-model calibration headroom at the declared start pose and
reports the retained count at startup. Every retained world and self barrier is
audited along the returned path.

This calibration should be regenerated and audited if the URDF, sphere fit,
fixed-joint reference, or desired physical self-clearance changes.
