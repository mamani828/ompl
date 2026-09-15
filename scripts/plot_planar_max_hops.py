"""Map certified hold times back to actual 2R arm poses along one ray."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hold_time import speed_capped_bound, speed_capped_time

out=Path(__file__).resolve().parents[1]/'results/planar_hold_bounds'
out.mkdir(parents=True,exist_ok=True)
L=np.array([1.,.8]); q0=np.deg2rad([30.,100.]); v=np.array([.8,-1.3])
d=.6; horizon=1.2
def fk(t):
    theta=np.cumsum(q0+t*v)
    seg=L[:,None]*np.column_stack((np.cos(theta),np.sin(theta)))
    return np.vstack(([0.,0.],np.cumsum(seg,axis=0)))
p0=fk(0)[-1]
seg=np.diff(fk(0),axis=0); w=np.cumsum(v)
u0=(w[:,None]*np.column_stack((-seg[:,1],seg[:,0]))).sum(axis=0)
S=float(np.linalg.norm(u0)); C=float(L@w**2); V=float(L@abs(w))
rate=float(L.sum()*abs(v[0])+L[1]*abs(v[1]))
t1=d/rate; t2=speed_capped_time(d,S,C,V)
ts=np.linspace(0,horizon,10001)
dist=np.array([np.linalg.norm(fk(t)[-1]-p0) for t in ts])
idx=np.flatnonzero(dist>=d)[0]
lo,hi=ts[idx-1],ts[idx]
for _ in range(50):
    mid=(lo+hi)/2
    if np.linalg.norm(fk(mid)[-1]-p0)<d: lo=mid
    else: hi=mid
tfk=(lo+hi)/2
assert t1<t2<tfk
assert abs(speed_capped_bound(t2,S,C,V)-d)<1e-12
assert np.max(dist[:idx])<=d
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,8))
ax.add_patch(Circle(p0,d,facecolor='#edf4ed',edgecolor='#759576',lw=1.8,ls='--'))
trail=np.array([fk(t)[-1] for t in np.linspace(0,tfk,300)])
ax.plot(*trail.T,color='#59615d',lw=1.5,zorder=2)
poses=[(0,'Start','#28343f','-',1.),(t1,'L1','#347db5','--',.85),
       (t2,'L2','#d18329','--',.85),(tfk,'FK budget limit','#238b70','-',.9)]
for t,label,color,ls,alpha in poses:
    arm=fk(t)
    ax.plot(*arm.T,'o',ls=ls,color=color,lw=3,ms=7,alpha=alpha,
            label=f'{label}: t = {t:.3f} s',zorder=3)
    ax.plot(*arm[-1],'o',color=color,ms=10,zorder=4)
offsets=[(55,-15),(65,-12),(65,0),(-160,30)]
for (t,label,color,_,_),offset in zip(poses,offsets):
    ax.annotate(label,fk(t)[-1],xytext=offset,textcoords='offset points',
                color=color,weight='bold',arrowprops=dict(arrowstyle='-',color=color),fontsize=13)
ax.annotate('Same clearance budget\n$d = 0.6$ m',p0+[-d,0],xytext=(-.62,.7),
            arrowprops=dict(arrowstyle='->',color='#759576'),color='#567557',fontsize=11)
ax.plot(0,0,'s',color='#28343f',ms=12)
ax.set(aspect='equal',xlim=(-.65,1.45),ylim=(-.15,2.1),xlabel='x [m]',ylabel='y [m]')
ax.grid(alpha=.15)
ax.legend(loc='lower left',fontsize=11,framealpha=.95)
ax.set_title('Maximum certified hop along the same C-space line\nL1 stops first; L2 lets the arm advance farther',fontsize=17,pad=16)
fig.text(.5,.035,'Each ghost is actual FK at that method’s stopping time: q₀ + t v.\n'
         '“FK budget limit” is the first exit from the clearance ball, not a collision or a whole-arm safety limit.',
         ha='center',fontsize=10)
fig.tight_layout(rect=(0,.085,1,1))
for ext in ['png','pdf']: fig.savefig(out/f'max_hop_ghosts.{ext}',dpi=190,bbox_inches='tight')
print(f'L1={t1:.6f}s, L2={t2:.6f}s, FK budget exit={tfk:.6f}s')
