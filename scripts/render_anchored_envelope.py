"""Compute the spatial Taylor tube underlying HoldTimeCertificate's IsoAnchor.

This illustrates the anchored world certificate alone, using its equations in
float64, not a full production HoldEngine hop (which takes max across terms).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from render_ur5_certificate_teaser import Model, BOX, HALF, ROOT, MARGIN
from teaser_l2 import ur5_rows
from hold_time import _anchored_prefix_bound

OUT=ROOT/'results/ur5_anchored_envelope';OUT.mkdir(parents=True,exist_ok=True)
m=Model();q=np.array([-.51,-1.05,1.3]);T=.9
# Same motion direction as the previous single-hop illustration.
old=json.loads((ROOT/'results/ur5_single_hop/audit.json').read_text());u=np.array(old['u'])
rows=ur5_rows(m,q);moving=np.flatnonzero(np.any(m.A>1e-12,axis=1))
records=[]
for g,h,n in rows:
    v0,a0,_,_=g.initial_motion(u[:n]);e=g.envelopes(u[:n],T)
    lo,hi=0.,T
    if _anchored_prefix_bound(T,v0,a0,e.H)<=h:lo=T
    else:
        # Match the production anchoredTime's conservative 12-step bisection.
        for _ in range(12):
            mid=(lo+hi)/2
            if _anchored_prefix_bound(mid,v0,a0,e.H)<=h:lo=mid
            else:hi=mid
    records.append((lo,v0,a0,e.H,h))
b=int(np.argmin([r[0] for r in records]));tau,v0,a0,H,h=records[b]
sphere=int(moving[b]);r=m.radii[sphere];c=m.fk(q)[1][sphere]
t=np.linspace(0,tau,401)
pred=c+t[:,None]*v0+.5*t[:,None]**2*a0
error=H*t**3/6
actual=np.array([m.fk(q+s*u)[1][sphere] for s in t])
residual=np.linalg.norm(actual-pred,axis=1)-error
assert residual.max()<1e-10
clearance=min(float(m.clearance(q+s*u).min()) for s in t)
assert clearance>=0
# Verify this Taylor containment for every moving collision sphere, too.
all_res=[]
for idx,(_,vi,ai,Hi,_) in zip(moving,records):
    c0=m.fk(q)[1][idx]
    real=np.array([m.fk(q+s*u)[1][idx] for s in t[::4]])
    nominal=c0+t[::4,None]*vi+.5*t[::4,None]**2*ai
    all_res.append(float(np.max(np.linalg.norm(real-nominal,axis=1)-Hi*t[::4]**3/6)))
assert max(all_res)<1e-10
teal='#007f89';orange='#df8b24';ink='#183f4b'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'mathtext.fontset':'stix','font.size':13})
fig=plt.figure(figsize=(11,4.8),facecolor='white')
# First panel: shape of the moving sphere-center tube, tightly zoomed.
ax=fig.add_axes([.07,.29,.38,.52]);ax.set_aspect('auto')
normal=v0/np.linalg.norm(v0);side=a0-normal*np.dot(a0,normal)
if np.linalg.norm(side)<1e-10:side=np.cross(normal,[0,0,1])
side/=np.linalg.norm(side);basis=np.array([normal,side])
curve=(pred-c)@basis.T;real=(actual-c)@basis.T
# A union of analytically sized projected disks. Grid sampling only renders it.
def tube(axis,curve,radii,color):
    lower=np.min(curve-radii[:,None],axis=0);upper=np.max(curve+radii[:,None],axis=0)
    pad=max(upper-lower)*.035+1e-6
    xx=np.linspace(lower[0]-pad,upper[0]+pad,600)
    yy=np.linspace(lower[1]-pad,upper[1]+pad,300)
    X,Y=np.meshgrid(xx,yy);field=np.full_like(X,np.inf)
    for pt,rho in zip(curve,radii):field=np.minimum(field,(X-pt[0])**2+(Y-pt[1])**2-rho*rho)
    axis.contourf(X,Y,field,levels=[-1e9,0],colors=[color],alpha=.22)
    axis.contour(X,Y,field,levels=[0],colors=[color],linewidths=1.6)
tube(ax,curve*1000,error*1000,orange)
ax.plot(curve[:,0]*1000,curve[:,1]*1000,color=orange,ls='--',lw=1.5,label='Taylor center')
ax.plot(real[:,0]*1000,real[:,1]*1000,color=teal,lw=2,label='Exact FK motion')
ax.scatter(*real[0]*1000,s=45,color=teal,zorder=5)
ax.scatter(*real[-1]*1000,s=45,color=teal,facecolors='white',zorder=5)
ax.set_xlabel('Along initial velocity (mm)');ax.set_ylabel('Transverse motion (mm)')
ax.set_title('(b) Computed kinematic envelope',color=ink,fontsize=17,pad=18)
ax.legend(loc='upper left',fontsize=10,frameon=False)
ax.text(.5,-.36,r'$p(t)\in B(p_0+t\dot p_0+\frac{1}{2}t^2\ddot p_0,\;Ht^3/6)$',
        transform=ax.transAxes,ha='center',fontsize=14)
# Second: same tube enlarged by physical sphere radius, against the actual box.
ax=fig.add_axes([.59,.23,.37,.62]);ax.set_aspect('equal')
tube(ax,pred[:,:2],r+error,orange)
ax.add_patch(Rectangle((BOX-HALF)[:2],2*HALF[0],2*HALF[1],fc='#c9c2b3',ec='#978f80'))
ax.add_patch(Circle(c[:2],r,fc=teal,ec=teal,alpha=.8))
ax.add_patch(Circle(actual[-1,:2],r,fc=teal,ec=teal,alpha=.20))
ax.plot(actual[:,0],actual[:,1],color=teal,lw=2)
ax.annotate('',actual[-1,:2],c[:2],arrowprops=dict(arrowstyle='->',color=teal,lw=2))
ax.annotate('Current',c[:2],xytext=(-60,-12),textcoords='offset points',color=teal,fontsize=12)
ax.annotate('Endpoint ghost',actual[-1,:2],xytext=(-95,35),textcoords='offset points',color=teal,fontsize=12,
            arrowprops=dict(arrowstyle='-',color=teal))
ax.text(BOX[0],BOX[1],'Actual\nobstacle',ha='center',va='center',color=ink)
ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)')
ax.set_title('(c) Entire sphere stays inside the tube',color=ink,fontsize=16,pad=18)
ax.text(.5,-.25,'Orange: Taylor tube + sphere radius',transform=ax.transAxes,ha='center',fontsize=12,color=orange)
for a in fig.axes:a.spines[['top','right']].set_visible(False)
fig.text(.5,.015,f'Anchored certificate: {tau*1000:.1f} ms hop | endpoint error radius {error[-1]*1000:.2f} mm | world margin {MARGIN*1000:.0f} mm',ha='center',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'bottom_row.{ext}',dpi=260)
original=Image.open('/home/ax-ml-ubu24/Downloads/Combined_MainFig.png')
combined=plt.figure(figsize=(10,10));a=combined.add_axes([0,.42,1,.58]);a.imshow(original);a.set_ylim(875,0);a.axis('off')
a=combined.add_axes([0,0,1,.43]);a.imshow(Image.open(OUT/'bottom_row.png'));a.axis('off')
combined.savefig(OUT/'figure1_mockup.png',dpi=220)
audit=dict(q=q.tolist(),u=u.tolist(),sphere=sphere,tau=tau,T=T,velocity=v0.tolist(),acceleration=a0.tolist(),H=H,
           endpoint_remainder_radius=float(error[-1]),max_sampled_taylor_violation=float(residual.max()),
           max_all_spheres_sampled_violation=max(all_res),min_sampled_clearance=clearance,
           code_reference='demos/HoldTimeCertificate.h: anchorEndpoint, prefixMaxNorm, anchoredTime',
           scope='Spatial Taylor tube underlying the anchored term, Python evaluation with production 12-step root rule. Not the max-of-terms production hop. Plots are orthographic projections of a sampled-time union; contour discretization is visual only.')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
