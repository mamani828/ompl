"""Exact 2R FK versus the planar specialization of HoldTimeCertificate.h."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

out = Path(__file__).resolve().parents[1] / 'results/planar_hold_bounds'
out.mkdir(parents=True, exist_ok=True)
L = np.array([1., .8])
q0 = np.deg2rad([30., 100.])
v = np.array([.8, -1.3])
T, d = 1.2, .6
t = np.linspace(0, T, 4001)
q = q0 + t[:, None]*v
theta = np.cumsum(q, axis=1)
w = np.cumsum(v)
links = L[None, :, None]*np.stack([np.cos(theta), np.sin(theta)], axis=-1)
p = links.sum(axis=1)
p0 = p[0]
rot = lambda x: np.stack([-x[..., 1], x[..., 0]], axis=-1)
vel = (w[None, :, None]*rot(links)).sum(axis=1)
acc = (-(w*w)[None, :, None]*links).sum(axis=1)
u0, a0 = vel[0], acc[0]
# Parallel, fixed planar axes: chi=A=W=0 and Omega_j=|sum_{k<=j} v_k|.
# Hence these are exactly the segment-sum V,C,H construction in buildFrame.
V = float(L @ np.abs(w))
C = float(L @ w**2)
H = float(L @ np.abs(w)**3)
S = np.linalg.norm(u0)
global_rate = (L.sum()*abs(v[0]) + L[1]*abs(v[1]))
local_rate = np.linalg.norm(p0)*abs(v[0])+L[1]*abs(v[1])
tc = (V-S)/C
speed = np.where(t <= tc, S*t+.5*C*t*t,
                 S*tc+.5*C*tc*tc+V*(t-tc))
quad = t[:, None]*u0+.5*t[:, None]**2*a0
anchored = np.maximum.accumulate(np.linalg.norm(quad, axis=1))+H*t**3/6
exact = np.maximum.accumulate(np.linalg.norm(p-p0, axis=1))
first_order = S*t+.5*C*t*t
curves = {'Global L1':global_rate*t, 'Local L1':local_rate*t,
          'Speed-capped':speed, 'Anchored':anchored}
def crossing(y):
    return float(np.interp(d, y, t)) if y[-1] >= d else None
times = {k: crossing(y) for k,y in curves.items()}
cert = max(T if x is None else x for x in times.values())
assert np.max(np.linalg.norm(acc, axis=1)) <= C+1e-12
assert np.all(exact <= speed+1e-12)
assert np.all(np.linalg.norm(p-p0-t[:, None]*u0, axis=1) <= .5*C*t*t+1e-12)
assert np.all(np.linalg.norm(p-p0-quad, axis=1) <= H*t**3/6+1e-12)
assert all(np.all(exact <= y+1e-12) for y in curves.values())
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axs = plt.subplots(2,2,figsize=(13,10))
blue, orange, green, purple = '#2474b5','#df8527','#239276','#8b5bb5'
ax=axs[0,0]
ax.plot(*q.T, color=blue, lw=2)
ax.scatter(*q[[0,-1]].T, color=[blue,orange], s=55)
ax.annotate('$q_0$',q[0],xytext=(8,5),textcoords='offset points')
ax.annotate('$q(T)$',q[-1],xytext=(-40,10),textcoords='offset points')
ax.set(title='A   The nominal joint motion is a straight line',xlabel='$q_1$ [rad]',ylabel='$q_2$ [rad]')
ax.text(.05,.08,'$q(t)=q_0+t v$\nJoint acceleration = 0',transform=ax.transAxes)
ax.grid(alpha=.2)

ax=axs[0,1]
for idx,alpha in zip([0,1000,2000,3000,4000],[1,.2,.3,.4,.8]):
    pts=np.vstack(([0,0], links[idx,0],p[idx]))
    ax.plot(*pts.T,'o-',color=blue if idx==0 else orange,alpha=alpha,lw=2)
ax.plot(*p.T,color='black',lw=2,label='Exact FK path')
ax.add_patch(Circle(p0,d,fill=False,ls='--',color=green,lw=2))
ax.plot(*p0,'o',color=blue,label='$p_0$')
ax.plot(*p[-1],'o',color=orange,label='$p(T)$')
ax.set(title='B   FK maps that line to a curved workspace path',xlabel='x [m]',ylabel='y [m]',aspect='equal')
ax.legend(fontsize=9,loc='lower left')
ax.text(.98,.96,'Dashed circle: displacement budget d = 0.6 m',ha='right',va='top',transform=ax.transAxes,fontsize=9)

ax=axs[1,0]
lin = p0+T*u0
pred = p0+quad[-1]
ax.add_patch(Circle(lin,.5*C*T*T,facecolor=orange,edgecolor=orange,alpha=.17))
ax.add_patch(Circle(pred,H*T**3/6,facecolor=purple,edgecolor=purple,alpha=.22))
ax.plot(*p.T,'k-',lw=2,label='Exact FK path')
ax.plot(*np.vstack([p0,lin]).T,'--',color=orange,label='Initial-velocity prediction')
ax.plot(*(p0+quad).T,':',color=purple,lw=2,label='Quadratic prediction')
ax.plot(*lin,'x',color=orange,ms=9)
ax.plot(*pred,'x',color=purple,ms=9)
ax.plot(*p[-1],'ko',label='Exact FK endpoint')
ax.set(title='C   What the bounds know about the endpoint at T',xlabel='x [m]',ylabel='y [m]',aspect='equal')
ax.text(.03,.97,'Orange radius: $CT^2/2$ = %.3f m\nPurple radius: $HT^3/6$ = %.3f m'%(.5*C*T*T,H*T**3/6),transform=ax.transAxes,va='top',fontsize=9)
ax.legend(loc='lower left',fontsize=8)

ax=axs[1,1]
for (label,y),color in zip(curves.items(),[blue,orange,green,purple]):
    ax.plot(t,y,color=color,lw=2,label=label)
    if times[label] is not None: ax.plot(times[label],d,'o',color=color)
ax.plot(t,first_order,'--',color=orange,alpha=.6,label='$St+Ct^2/2$ (uncapped)')
ax.plot(t,exact,'k',lw=2.5,label='Exact FK prefix displacement')
ax.axhline(d,color='gray',ls='--')
ax.axvline(cert,color=purple,ls=':',alpha=.6)
ax.set(title='D   Same motion, same clearance: when does each stop?',xlabel='t [s]',ylabel='Maximum displacement / bound [m]',ylim=(0,1.45))
ax.legend(fontsize=8,loc='upper left')
ax.grid(alpha=.15)
fig.suptitle('Two-link planar arm: our kinematic bounds versus exact FK',fontsize=18,weight='bold')
fig.text(.5,.025,'Lengths = (1.0, 0.8) m   •   q₀ = (30°, 100°)   •   v = (0.8, −1.3) rad/s   •   T = 1.2 s\n'
         'End-effector displacement only; the clearance circle is a hypothetical world-safety budget, not whole-arm collision checking.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.075,1,.95),h_pad=2.2)
for ext in ['png','pdf']: fig.savefig(out/f'planar_hold_bounds.{ext}',dpi=180,bbox_inches='tight')
summary=dict(V=V,C=C,H=H,S=float(S),q0=q0.tolist(),v=v.tolist(),T=T,
             p0=p0.tolist(),pT=p[-1].tolist(),exact_prefix_at_T=float(exact[-1]),
             bound_at_T={k:float(y[-1]) for k,y in curves.items()},
             certificate_times=times,combined_certificate=cert,FK_budget_exit=crossing(exact))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
