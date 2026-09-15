"""Illustrative geometry of hold certificates; synthetic, not robot measurements."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parents[1] / 'results/hold_bound_geometry'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
blue, orange, green, purple = '#2474b5', '#df8527', '#239276', '#8b5bb5'

ax = axs[0, 0]
ax.set_title('A   Clearance is a displacement budget', loc='left', weight='bold')
ax.axvspan(1, 1.7, color='#e8c4c4')
ax.axvline(1, color='#a34b4b', lw=2)
ax.text(1.08, .85, 'Inflated\nobstacle', color='#924343')
ax.add_patch(Circle((0, 0), 1, facecolor='#eaf4fa', edgecolor=blue, lw=2))
ax.add_patch(Circle((0, 0), .58, fill=False, edgecolor=orange, lw=2, ls='--'))
ax.plot(0, 0, 'o', color=blue)
ax.annotate('', (1, 0), (0, 0), arrowprops=dict(arrowstyle='<->', color=blue))
ax.text(.42, .08, '$d$')
ax.text(-.88, -.78, 'Safe ball about $p_0$', color=blue)
ax.text(-.52, .35, '$B(t)$', color=orange)
ax.set(xlim=(-1.2, 1.7), ylim=(-1.15, 1.15), aspect='equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('Center motion stays safe while its bound is ≤ d.\nObstacle inflation includes the robot sphere radius.')

ax = axs[0, 1]
ax.set_title('B   Different bounds spend that budget differently', loc='left', weight='bold')
t = np.linspace(0, 2.5, 800)
S, C, V, H = .22, .5, 1.05, .24
tc = (V-S)/C
speed = np.where(t <= tc, S*t+.5*C*t*t,
                 S*tc+.5*C*tc*tc+V*(t-tc))
# A consistent illustrative quadratic prediction: initial acceleration turns
# against the velocity; its magnitude is below C, and its jerk is zero <= H.
u = np.array([S, 0.]); a = np.array([-.08, .20])
pred = t[:, None]*u + .5*t[:, None]**2*a
anchor = np.maximum.accumulate(np.linalg.norm(pred, axis=1))+H*t**3/6
curves = [('Global L1', 1.35*t, blue), ('Local L1', 1.05*t, orange),
          ('Speed-capped', speed, green), ('Anchored', anchor, purple)]
for label, y, color in curves:
    ax.plot(t, y, color=color, lw=2.3, label=label)
    root = np.interp(1., y, t)
    ax.plot(root, 1, 'o', color=color)
    ax.vlines(root, 0, 1, color=color, alpha=.35, ls=':')
ax.axhline(1, color='#a34b4b', ls='--')
ax.text(.04, 1.06, 'clearance d', color='#924343')
ax.set(xlim=(0, 2.5), ylim=(0, 1.5), xlabel='Held-motion parameter t', ylabel='Displacement bound / clearance')
ax.legend(loc='upper left', fontsize=9)
ax.text(.97, .06, 'Crossing = certificate expires\nOrdering shown is illustrative', transform=ax.transAxes,
        ha='right', fontsize=10)

ax = axs[1, 0]
ax.set_title('C   Anchored: a parabola inside an error tube', loc='left', weight='bold')
ss = np.linspace(0, 1.8, 150)
uu, aa, hh = np.array([.5, .18]), np.array([-.15, .35]), .23
pp = ss[:, None]*uu+.5*ss[:, None]**2*aa
ax.add_patch(Circle((0, 0), 1.5, facecolor='#f0f6fa', edgecolor=blue, ls='--'))
for s in np.linspace(.3, 1.8, 9):
    p = s*uu+.5*s*s*aa
    ax.add_patch(Circle(p, hh*s**3/6, facecolor=purple, edgecolor=purple, alpha=.13))
ax.plot(pp[:, 0], pp[:, 1], color=purple, lw=2.5)
ax.plot(0, 0, 'o', color=blue)
ax.annotate('Quadratic prediction\n$s u_0 + s^2 a_0/2$', xy=pp[85], xytext=(-1.3, .6),
            arrowprops=dict(arrowstyle='->', color=purple), color=purple)
ax.annotate('Error radius\n$H s^3/6$', xy=pp[-1]+[.20, .05], xytext=(1.03, .15),
            arrowprops=dict(arrowstyle='->', color=purple), color=purple)
ax.text(-1.3, -.68, 'The entire tube must fit\ninside the clearance ball.', color=blue)
ax.set(aspect='equal', xlim=(-1.65, 1.85), ylim=(-.95, 1.75))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('Maximize over every s ∈ [0, t]: a safe endpoint is insufficient.')

ax = axs[1, 1]
ax.set_title('D   Self-collision: sideways motion can be free', loc='left', weight='bold')
ax.add_patch(Circle((0, 0), 1, facecolor='#e8c4c4', edgecolor='#a34b4b', lw=2))
ax.axvline(1, color=green, lw=2)
ax.axvspan(1, 2.8, color=green, alpha=.07)
p0 = np.array([1.7, 0])
ax.add_patch(Circle(p0, .7, fill=False, edgecolor=blue, ls='--', lw=2))
ax.plot(*p0, 'o', color=blue)
ax.annotate('', (1, 0), p0, arrowprops=dict(arrowstyle='<->'))
ax.text(1.27, -.19, '$d$')
ax.annotate('', (1.7, 1.35), p0, arrowprops=dict(arrowstyle='->', color=purple, lw=3))
ax.text(1.83, .86, 'Tangential\nmotion', color=purple)
ax.text(-.68, -.1, 'Forbidden\nrelative positions\n$\\|r\\| < r_a+r_b$', fontsize=10)
ax.text(1.06, -1.13, 'Fixed separating plane', color=green, fontsize=10)
ax.text(1.87, -.56, 'Isotropic\nbudget', color=blue, fontsize=10)
ax.set(aspect='equal', xlim=(-1.2, 2.85), ylim=(-1.3, 1.5))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('Normal projection charges only closing motion;\nthe remainder still accounts for uncertain curvature.')

fig.suptitle('How long can we safely hold a joint-space direction?', fontsize=19, weight='bold', y=.98)
fig.text(.5, .025, 'For each collision constraint: take the longest valid certificate.  '
         'Across constraints: take the shortest, capped by the horizon.\n'
         'Synthetic illustrations • Certificate expiry is not predicted collision.',
         ha='center', fontsize=11)
fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2.5)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'hold_bound_geometry.{ext}', dpi=200, bbox_inches='tight')
print(OUT / 'hold_bound_geometry.png')
