# Reachy2 bimanual OMPL + CBF interface

`demo_Reachy2CBFPlanning` plans the two Reachy2 arms together in a 14-dimensional
joint space. The left and right goals are Cartesian arm-tip positions. A
damped-least-squares IK stage converts them into a coupled joint-space goal,
then OMPL control RRT plans with every propagation step projected through a
qpmad control-barrier-function QP.

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

# The third argument is CBF rope-shortcut anchor spacing in radians.
# Zero disables shortcutting; 0.05 is the aggressive default.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_raw.path 0
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_short.path 0.05

# Custom left and right Cartesian targets.
./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path \
    0.62 0.20 1.1344  0.62 -0.20 1.1344
```

The path file begins with the ordered joint names and then contains one
raw-radian 14-joint configuration per line. Because the requested goal is
Cartesian, an OMPL path that is approximate relative to one IK representative
is accepted only when both actual tips finish within 4 cm of their requested
positions. The executable returns nonzero when IK fails, that Cartesian test
fails, or the dense post-plan barrier audit fails.

Shortcutting uses the dimension-independent `ompl::cbf::ropeShortcut` overload.
Candidates are evaluated by complete Reachy CBF rollouts rather than straight
joint-space chords, and the result retains every filtered integration waypoint.
The demo uses 0.05-radian anchors and accepts improvements down to 1% of that
spacing. This is more aggressive than the generic rope defaults while preserving
the same filtered-rollout and endpoint-arrival safety requirements.
The demo reports length before/after, accepted and attempted rollouts, runtime,
and the largest endpoint gap introduced by arrival snapping.

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

The demo uses an all-valid OMPL state checker deliberately: environment and
self-collision safety are imposed in the propagator by the CBF, matching the
existing CBF architecture in this repository.

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
