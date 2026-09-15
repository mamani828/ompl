# Corrected L2 contribution figure

| file | contents |
|---|---|
| `combined.pdf` | all three panels, one plate |
| `ur5_adaptive.pdf` | (a) UR5 alone |
| `planar_cspace.pdf` | (b) and (c) side by side |
| `planar_workspace.pdf` | (b) alone |
| `planar_config.pdf` | (c) alone |

Every file also has a matching 300 dpi PNG. `simulator_l2.png` is the raytraced
plate panel (a) is drawn over.

## Typesetting

Text is set in Computer Modern so the figure matches the document rather than
looking pasted in. `latex_fonts()` prefers a real TeX run -- identical glyphs,
kerning and math spacing -- and needs `latex` and `dvipng` on PATH; when those
are absent it falls back to matplotlib's *bundled* `cmr10`/`cmmi10`/`cmsy10`,
which are the same typefaces laid out by matplotlib instead of TeX. The script
prints which path it took. Fonts are embedded subset (`pdf.fonttype` 42), so the
PDFs render identically anywhere.

Two consequences worth knowing if you move labels: point offsets (`xytext` with
`textcoords='offset points'`) were re-tuned for Computer Modern, which sets
wider than the previous sans, and `axes.formatter.use_mathtext` is required
because `cmr10` has no minus glyph -- without it negative ticks render as
hyphens.

## Text in the plate

Deliberately minimal: panel (a) carries two callouts and nothing else. What a
dot and a faded arm mean lives in the caption below, where it costs no space in
the figure. Sizes are in the `FS` dict at the top of the script, set for
two-column print rather than on-screen viewing.

Joint labels in (b) are anchored to the joints they name with `annotate` at
`zorder=9`, not placed at fixed coordinates: `arm()` fills capsules at zorder
4-5 and joints at 6 while text defaults to 3, so a fixed label is painted over
as soon as the pose changes.

Suggested caption:

**L2-envelope-certified adaptive integration.** (a) A UR5 follows an
obstacle-avoiding path with integration endpoints selected by the L2 hold-time
certificate. Dots show their end-effector FK positions; the faded arm shows
the previous configuration. (b) A planar two-link arm illustrates one certified
hop. (c) The L2 radial envelope (teal) extends beyond the global L1 reference
(dashed). The selected displacement lies outside L1 but inside L2. Beige
denotes configurations in collision with the workspace obstacles.

Both panels call `hold_time.level2`. There is no L1 fallback or ellipse
substituted for the L2 region. The point bound integrates `min(S + C t, V)`,
where `S = ||J(q) v||` and `C, V` are valid acceleration and speed envelopes.

The planar region is sampled along 360 unit joint-space rays with horizon 1.5.
Each radius is the minimum L2 hold over the relevant link endpoints. It is a
radial region, not assumed convex. Bounding both endpoints bounds every link
centerline point by affine interpolation; hence the entire capsule is covered.
Clearances and the collision map use exact capsule-circle separation.

Planar links have lengths 0.9 and 0.7 m and radius 0.045 m. The dashed L1
reference uses the same clearances and global matrix `[[0.9, 0], [1.6, 0.7]]`.
The selected ray exposes cancellation from opposite joint rotations. Its L2
limit is 0.63309, versus 0.20735 for L1. The executed hop uses 88% of L2:
**2.6868 times the L1 limit**. This is an illustrative example, not an aggregate
performance result.

The UR5 geometric route is retained from the earlier RRT illustration and
resampled with pure L2 holds for all moving URDF collision spheres against the
obstacle. Joint axes and segment geometry are rebuilt at each endpoint. The
image is re-rendered at the new L2 endpoint. Faint branches remain the original
exploration geometry; this is not a new L2 planner benchmark run. These are
environment-clearance illustrations, not mesh-level or self-collision proofs.

`audit.json` records configurations, ray boundaries, hold intervals, and
clearance checks. Validation includes dense UR5 and planar path checks,
21,960 sampled points along the planar L2 rays, and an explicit check that the
chosen hop exceeds L1 while remaining inside L2.

Regenerate from the repository root:

```sh
.venv_mbm/bin/python scripts/render_adaptive_multiscale_teaser.py
```

The L2 geometry and rendering helpers are in `scripts/teaser_l2.py`.
