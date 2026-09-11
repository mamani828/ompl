# UR5 certificate teaser

`teaser.pdf` / `teaser.png` are the selected, labeled paper composition.
`teaser_clean.pdf` / `teaser_clean.png` have no labels or hop arrow.
`teaser_0`, `teaser_1`, and `teaser_2` retain the wider camera alternatives.
`simulator_*.png` are the PyBullet image plates before vector overlays.

Suggested caption:

**Certified adaptive integration on a UR5.** The translucent arm denotes the
current configuration and the solid arm the endpoint of a certified integration
step. The shaded surface depicts a three-joint slice of the joint-space
certificate mapped through end-effector forward kinematics. The highlighted
hop remains inside its originating joint-space certificate. Dots mark the
end-effector positions at integration endpoints; faint branches show the
exploration tree. Larger certified steps in open space become shorter as the
arm approaches the obstacle.

Geometry and provenance:

- Actual UR5 meshes and transformations from the repository's URDF; no generated imagery.
- Reproducible illustrative RRT in the first three joints, with wrist angles fixed.
  The highlighted route is a certified shortcut of the RRT solution.
- All URDF collision spheres contribute environment-clearance rows. Each row
  uses a conservative global serial-chain reach bound derived from URDF joint
  offsets and the sphere's local center. The margin is 15 mm.
- The depicted polytope satisfies `A |delta q| <= h(q)` and a local chart bound
  `|delta q_j| <= 0.48 rad`. Its facets and edges are sampled through exact FK.
  It is **not** a Cartesian convex safe set or a certificate based only on
  end-effector clearance. A connecting Cartesian chord is not certified.
- The shaded surface is a tessellated visualization of the nonlinear FK image.
- This illustration certifies environment clearance for the sphere model;
  it does not implement a self-collision certificate or establish exact mesh
  safety. It is not a recorded MotionBenchMaker/CBF benchmark experiment.
- `geometry.json` records configurations, hop endpoints, certificate vertices,
  utilization, dense clearance audit, and independent PyBullet FK agreement.

Regenerate from the repository root:

```sh
.venv_mbm/bin/python scripts/render_ur5_certificate_teaser.py
```
