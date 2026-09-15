# Certified steering-horizon explainer

This publication figure is computed from one two-link planar arm, one circular obstacle,
and one held command. The configuration-space panel shows the actual radial boundaries
of both certificates, evaluated over 721 control directions, together with the exact
configuration obstacle of the spherical end-effector row. The final panel
plots the jointwise reach-radius bound and the integrated motion-dependent speed
envelope for that same motion.

Every point in either star-shaped set is joined to $q_0$ by a segment certified using that
direction. Numerical parameters, derived constants, and crossing times are recorded in
`audit.json`.

Regenerate with:

```sh
python3 scripts/plot_certificate_horizon_explainer.py
```

`figure.tex` contains the region definitions, figure environment, caption, and
the explanatory paragraph that follows the figure.

Suggested caption: **Comparison of certified steering horizons.** The jointwise
Lipschitz certificate independently bounds the workspace displacement induced by each
joint, yielding the conservative horizon $\tau_{\rm lip}$. The motion-dependent
certificate bounds displacement along the particular held-control trajectory, yielding
the larger horizon $\tau_{\rm disp}$. Both certify the same continuous trajectory segment
without intermediate collision evaluation.
