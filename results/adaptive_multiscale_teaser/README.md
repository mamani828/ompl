# Adaptive integration and configuration-space certificates

**Superseded L1 draft.** The corrected contribution figure is in
`../adaptive_l2_teaser/`. The generator now produces that L2 version;
these older files are retained as a visual draft.

Exports:

- `combined.pdf` / `combined.png`: UR5 above planar workspace and C-space.
- `ur5_adaptive.pdf` / `ur5_adaptive.png`: standalone UR5 without a certificate surface.
- `planar_cspace.pdf` / `planar_cspace.png`: standalone planar workspace and C-space.

The earlier shaded UR5 teaser remains in `../ur5_certificate_teaser/`.

Suggested caption:

**Adaptive integration and configuration-space certification.** (a) A UR5
traverses an obstacle-avoiding path with larger integration steps in open space
and smaller steps near the obstacle. Each dot is an actual FK-evaluated
integration endpoint; amber identifies short joint-space steps. The translucent
arm shows the previous pose. (b) A planar two-link arm illustrates one certified
hop, from the faded configuration to the solid configuration. (c) The same
arm's configuration space: beige denotes collision configurations, while the
polygons show the current certificate and the smaller certificate recomputed
after the hop. The connecting displacement is inside the initial certificate.

Technical details:

- UR5 data and screenshot come from the existing reproducible UR5 illustration.
  No dots are manually repositioned or subsampled. Amber marks steps below
  the 38th percentile of joint-space step length. The spatial distance between
  FK endpoints is also affected by the robot Jacobian; it is not a time axis.
- The planar links are exact capsules, lengths 0.9 m and 0.7 m, radius 0.045 m.
  The two obstacles are circles. The plotted collision boundary is a numerical
  contour of exact capsule-circle separation on a dense joint-angle grid.
- With per-link clearance `h(q)`, the certificate is
  `C(q) = {q + delta : A |delta| <= h(q)}`,
  where `A = [[0.9, 0], [1.6, 0.7]]`. These global displacement bounds apply
  to all points of each link. The polygons are actual halfspace intersections.
- Both panels use illustrative environment-clearance certificates, not new
  benchmark evidence. The UR5 scope remains the documented sphere model;
  no new self-collision certificate is claimed.
- `audit.json` includes certificate vertices, hop utilization, and sampled
  region / dense-path clearance checks. Polygon membership is checked
  algebraically; sampling supplements the conservative certificate.

Regenerate from the repository root:

```sh
.venv_mbm/bin/python scripts/render_adaptive_multiscale_teaser.py
```

This uses the saved `results/ur5_certificate_teaser/simulator_0.png` plate.
Regenerate that plate with `scripts/render_ur5_certificate_teaser.py` if needed.
