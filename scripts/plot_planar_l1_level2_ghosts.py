"""Two-link FK ghosts with actual L1 and speed-capped Level-2 envelopes."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hold_time import speed_capped_bound

out = Path(__file__).resolve().parents[1]/'results/planar_hold_bounds'
out.mkdir(parents=True, exist_ok=True)
L=np.array([1.,.8]); q0=np.deg2rad([30.,100.]); v=np.array([.8,-1.3]); T=1.2
def fk(t):
    th=np.cumsum(q0+t*v)
    seg=L[:,None]*np.column_stack((np.cos(th),np.sin(th)))
    return np.vstack(([0.,0.],np.cumsum(seg,axis=0)))
p0=fk(0)[-1]; w=np.cumsum(v)
seg=np.diff(fk(0),axis=0)
u0=(w[:,None]*np.column_stack((-seg[:,1],seg[:,0]))).sum(axis=0)
S=float(np.linalg.norm(u0)); C=float(L@w**2); V=float(L@abs(w))
rate=float(L.sum()*abs(v[0])+L[1]*abs(v[1]))
radii=[rate*T,speed_capped_bound(T,S,C,V)]
ts=np.linspace(0,T,1001); path=np.array([fk(t)[-1] for t in ts])
for t,p in zip(ts,path):
    assert np.linalg.norm(p-p0)<=min(rate*t,speed_capped_bound(t,S,C,V))+1e-12
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13,7))
colors=['#357fba','#d18530']
for ax,r,color,title in zip(axs,radii,colors,['L1: worst-case lever arms','Level 2: initial speed + acceleration cap']):
    ax.add_patch(Circle(p0,r,facecolor=color,alpha=.10,edgecolor='none'))
    ax.add_patch(Circle(p0,r,fill=False,edgecolor=color,lw=2.3))
    for t in [.3,.6,.9]:
        arm=fk(t)
        ax.plot(*arm.T,'o-',color='#657987',alpha=.18,lw=3,ms=5)
    ax.plot(*fk(0).T,'o-',color='#253847',lw=3,ms=6,zorder=4)
    ax.plot(*fk(T).T,'o--',color='#657987',lw=3,ms=6,alpha=.75,zorder=4)
    ax.plot(*path.T,color='black',lw=1.5,zorder=5)
    ax.plot(*p0,'o',color=color,ms=8,zorder=6)
    ax.annotate('$p_0$',p0,xytext=(-25,-20),textcoords='offset points')
    ax.annotate('Exact FK at T',fk(T)[-1],xytext=(-105,22),textcoords='offset points',
                arrowprops=dict(arrowstyle='->',color='#657987'),fontsize=11)
    angle=-.52
    end=p0+r*np.array([np.cos(angle),np.sin(angle)])
    ax.annotate('',end,p0,arrowprops=dict(arrowstyle='->',color=color,lw=1.6))
    mid=(p0+end)/2
    ax.text(mid[0]+.08,mid[1]-.24,f'{r:.3f} m',color=color,weight='bold')
    ax.set(title=title,aspect='equal',xlim=(-2.9,3.65),ylim=(-2.1,4.4),xlabel='x [m]')
    ax.grid(alpha=.12)
axs[0].set_ylabel('y [m]')
axs[0].text(.04,.96,'$R_1(T)=T[(L_1+L_2)|v_1|+L_2|v_2|]$',transform=axs[0].transAxes,va='top',fontsize=11)
axs[1].text(.04,.96,'$R_2(T)=\\int_0^T\\min(S+Cs,V)\\,ds$\n'
            '$S=0.955$ m/s,  $C=0.840$ m/s²\n$V=1.200$ m/s',transform=axs[1].transAxes,va='top',fontsize=11)
fig.suptitle('Same held motion, same final arm — two end-effector bounds',fontsize=18,weight='bold',y=.98)
fig.legend(handles=[Line2D([0],[0],color='#253847',lw=3,label='Starting arm'),
                    Line2D([0],[0],color='#657987',lw=3,ls='--',label='Exact final arm (ghost)'),
                    Line2D([0],[0],color='black',lw=1.5,label='Exact end-effector path')],
           loc='lower center',bbox_to_anchor=(.5,.085),ncol=3,frameon=False,fontsize=11)
fig.text(.5,.035,'T = 1.2 s  •  q(t) = (30°, 100°) + t (0.8, −1.3) rad/s  •  Links: 1.0 m, 0.8 m\n'
         'Shaded disks enclose end-effector displacement for the whole hold; they are not reachable arm configurations.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.14,1,.95))
for ext in ['png','pdf']: fig.savefig(out/f'l1_level2_ghosts.{ext}',dpi=190,bbox_inches='tight')
print(f'L1 radius: {radii[0]:.6f} m; Level-2 radius: {radii[1]:.6f} m; FK displacement: {np.linalg.norm(path[-1]-p0):.6f} m')

# Overlay of the same deterministic FK motion and both conservative disks.
fig2, ax = plt.subplots(figsize=(9, 9))
for r, color, label in zip(radii, colors, ['L1', 'Level 2']):
    ax.add_patch(Circle(p0, r, facecolor=color, alpha=.12, edgecolor='none'))
    ax.add_patch(Circle(p0, r, fill=False, edgecolor=color, lw=2.5,
                        label=f'{label} enclosure: {r:.3f} m'))
for t in [.3, .6, .9]:
    ax.plot(*fk(t).T, 'o-', color='#657987', alpha=.18, lw=3, ms=5)
ax.plot(*fk(0).T, 'o-', color='#253847', lw=3, ms=6, label='Starting arm')
ax.plot(*fk(T).T, 'o--', color='#657987', lw=3, ms=6, alpha=.8,
        label='Exact final arm (FK)')
ax.plot(*path.T, color='black', lw=2.5, label='Exact FK end-effector path', zorder=6)
ax.plot(*p0, 'o', color='#253847', ms=8, zorder=7)
ax.plot(*path[-1], '*', color='#181818', ms=16, zorder=8)
ax.annotate('Initial endpoint $p_0$', p0, xytext=(35, -30),
            textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=11)
ax.annotate('FK endpoint at T\n0.761 m from $p_0$', path[-1], xytext=(-150, 35),
            textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=12)
for r, angle, color, label in zip(radii, [-.55, -.95], colors, ['L1', 'L2']):
    end = p0+r*np.array([np.cos(angle), np.sin(angle)])
    ax.annotate('', end, p0, arrowprops=dict(arrowstyle='->', color=color, lw=1.7))
    pos = p0+.78*r*np.array([np.cos(angle), np.sin(angle)])
    ax.text(pos[0]+.08, pos[1]-.05, f'{label}\n{r:.3f} m', color=color,
            fontsize=12, weight='bold', va='top')
ax.set(aspect='equal', xlim=(-2.9, 3.65), ylim=(-2.1, 4.4), xlabel='x [m]', ylabel='y [m]')
ax.grid(alpha=.12)
ax.legend(loc='upper right', fontsize=10, framealpha=.95)
ax.set_title('Exact FK vs. L1 vs. Level 2\nOne two-link motion, T = 1.2 s', fontsize=18, pad=16)
fig2.text(.5, .025, 'FK gives the actual path and endpoint. L1 and L2 give conservative displacement disks.\n'
          'Both disks are centered at the initial endpoint; shaded area does not mean every point is reachable.',
          ha='center', fontsize=10)
fig2.tight_layout(rect=(0, .065, 1, 1))
for ext in ['png', 'pdf']:
    fig2.savefig(out/f'l1_level2_fk_overlay.{ext}', dpi=190, bbox_inches='tight')
