"""Computed limiting-sphere certificate image against the teaser box."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from render_ur5_certificate_teaser import Model, BOX, HALF, ROOT, MARGIN
from teaser_l2 import ur5_rows
from hold_time import level2

out=ROOT/'results/ur5_binding_obstacle';out.mkdir(parents=True,exist_ok=True)
m=Model(); candidates=[np.array([x,-1.05,1.3]) for x in np.linspace(-1.3,.5,181)]
q=min((q for q in candidates if m.clearance(q).min()>.04),key=lambda q:abs(m.clearance(q).min()-.055))
rows=ur5_rows(m,q); angles=np.linspace(0,2*np.pi,361)
u=np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]
times=np.array([[level2(g,v[:n],h,.9).tau for g,h,n in rows] for v in u])
tau=times.min(axis=1); ray=int(np.argmin(tau))
moving=np.flatnonzero(np.any(m.A>1e-12,axis=1)); sphere=int(moving[times[ray].argmin()])
fractions=np.linspace(0,1,31)
centers=np.array([[m.fk(q+f*t*v)[1] for f in fractions] for t,v in zip(tau,u)])
grid=centers[:,:,sphere]; c=m.fk(q)[1][sphere];r=m.radii[sphere]
closest=np.clip(c,BOX-HALF,BOX+HALF);normal=(closest-c)/np.linalg.norm(closest-c)
surface=c+r*normal
clearance=min(float(m.clearance(q+f*t*v).min()) for t,v in zip(tau,u) for f in fractions)
assert clearance>=-1e-9
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'mathtext.fontset':'stix','font.size':14})
teal='#007f89';orange='#df8b24';ink='#183f4b'
fig=plt.figure(figsize=(10,4.2),facecolor='white')
ax=fig.add_axes([.06,.18,.40,.68]); ax.set_aspect('equal')
# Side projection shows all collision sphere centers and their actual radii.
for j,alpha in [(0,.55),(15,.18),(30,.10)]:
    cs=centers[ray,j]
    for ci,ri in zip(cs,m.radii):ax.add_patch(Circle(ci[[1,2]],ri,fc=teal,ec='none',alpha=alpha))
ax.add_patch(Rectangle((BOX-HALF)[[1,2]],2*HALF[1],2*HALF[2],fc='#c9c2b3',ec='#978f80',alpha=.8))
ax.plot(grid[:,-1,1],grid[:,-1,2],color=teal,lw=2)
ax.scatter(c[1],c[2],color=orange,s=55,zorder=9)
ax.annotate('Limiting sphere',c[[1,2]],xytext=(-60,30),textcoords='offset points',color=orange,
            arrowprops=dict(arrowstyle='->',color=orange),fontsize=12)
ax.autoscale_view();ax.set_xlabel('$y$ (m)');ax.set_ylabel('$z$ (m)')
ax.set_title('(b) UR5 collision geometry + ghosts',color=ink,fontsize=15)
ax=fig.add_axes([.59,.18,.37,.68]);ax.set_aspect('equal')
ax.add_patch(Rectangle((BOX-HALF)[:2],2*HALF[0],2*HALF[1],fc='#c9c2b3',ec='#978f80'))
ax.fill(grid[:,-1,0],grid[:,-1,1],color=teal,alpha=.16)
ax.plot(grid[:,-1,0],grid[:,-1,1],color=teal,lw=2)
for k in range(0,360,30):ax.plot(grid[k,:,0],grid[k,:,1],color=teal,alpha=.25,lw=.7)
for j in [0,15,30]:ax.add_patch(Circle(grid[ray,j,:2],r,fc=teal,ec=teal,alpha=.18))
ax.plot(grid[ray,:,0],grid[ray,:,1],color=orange,lw=2)
ax.scatter(*c[:2],color=orange,s=35,zorder=9)
ax.annotate('',closest[:2],surface[:2],arrowprops=dict(arrowstyle='<->',color=orange,lw=1.8))
ax.annotate('Initial clearance',((closest+surface)/2)[:2],xytext=(15,-30),textcoords='offset points',color=orange,fontsize=11)
ax.text(BOX[0],BOX[1],'Actual\nbox',ha='center',va='center',color=ink)
ax.autoscale_view();ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)')
ax.set_title('(c) Sphere motion against the obstacle',color=ink,fontsize=15)
for a in fig.axes:a.spines[['top','right']].set_visible(False)
fig.text(.5,.025,r'Computed FK image of certified motions: $D_i(\tau;u)\leq h_i$ for every collision sphere',ha='center',fontsize=13)
for ext in ['png','pdf','svg']:fig.savefig(out/f'bottom_row.{ext}',dpi=250)
original=Image.open('/home/ax-ml-ubu24/Downloads/Combined_MainFig.png')
combined=plt.figure(figsize=(10,10));ax=combined.add_axes([0,.41,1,.59]);ax.imshow(original);ax.set_ylim(875,0);ax.axis('off')
ax=combined.add_axes([0,0,1,.42]);ax.imshow(Image.open(out/'bottom_row.png'));ax.axis('off')
combined.savefig(out/'figure1_mockup.png',dpi=200)
audit=dict(q=q.tolist(),sphere=sphere,link=m.spheres[sphere][0],radius=float(r),
           closest_box_point=closest.tolist(),binding_hold=float(tau[ray]),
           margin=MARGIN,min_sampled_clearance=clearance,directions=u.tolist(),holds=tau.tolist(),
           scope='Python Level-2, world collision, q1/q2 slice. Curved contour is sampled FK image of certified sphere-center motions, not a Cartesian enclosing bound. Circles are physical sphere projections.')
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print({k:audit[k] for k in ['q','sphere','link','binding_hold','min_sampled_clearance']})
