"""Current pose, actual held-control endpoint, and scalar worst-case witness."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
import pybullet as p
from render_ur5_certificate_teaser import Model, BOX, HALF, ROOT, URDF, JOINTS, WRIST, MARGIN
from teaser_l2 import ur5_rows
from hold_time import level2, speed_capped_bound

OUT=ROOT/'results/ur5_single_hop';OUT.mkdir(parents=True,exist_ok=True)
m=Model();q=np.array([-.51,-1.05,1.3]);rows=ur5_rows(m,q)
angles=np.linspace(0,2*np.pi,360,endpoint=False)
directions=np.c_[np.cos(angles),np.sin(angles),np.zeros(360)]
holds=np.array([[level2(g,v[:n],h,.9).tau for g,h,n in rows] for v in directions])
a,b=np.unravel_index(holds.argmin(),holds.shape);tau=float(holds[a,b]);u=directions[a]
sphere=int(np.flatnonzero(np.any(m.A>1e-12,axis=1))[b]);r=float(m.radii[sphere])
g,h,n=rows[b];e=level2(g,u[:n],h,.9).envelopes
D=speed_capped_bound(tau,float(np.linalg.norm(g.J0@u[:n])),e.C,e.V)
ts=np.linspace(0,tau,401);cs=np.array([m.fk(q+t*u)[1][sphere] for t in ts]);c=cs[0]
closest=np.clip(c,BOX-HALF,BOX+HALF);normal=(closest-c)/np.linalg.norm(closest-c)
worst=c+D*normal
assert abs(D-h)<1e-10
assert np.linalg.norm(cs-c,axis=1).max()<=D+1e-10
clearance=min(float(m.clearance(q+t*u).min()) for t in ts)
assert clearance>=-1e-10
p.connect(p.DIRECT);body=p.loadURDF(str(URDF),useFixedBase=True)
ids={p.getJointInfo(body,j)[1].decode():j for j in range(p.getNumJoints(body))}
for j in range(p.getNumJoints(body)):
    name=p.getJointInfo(body,j)[12].decode()
    p.changeVisualShape(body,j,rgbaColor=([.08,.49,.55,1] if 'shoulder' in name or 'wrist' in name else [.68,.74,.77,1]))
shape=p.createVisualShape(p.GEOM_BOX,halfExtents=HALF,rgbaColor=[.70,.67,.60,1])
p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=BOX)
w,ht=1600,1300
view=p.computeViewMatrix([2.5,-3.2,2.7],[.05,.34,1.22],[0,0,1])
projection=p.computeProjectionMatrixFOV(22,w/ht,.05,10)
M=np.array(projection).reshape(4,4,order='F')@np.array(view).reshape(4,4,order='F')
def project(points):
    points=np.atleast_2d(points);v=np.c_[points,np.ones(len(points))]@M.T
    return np.c_[(v[:,0]/v[:,3]+1)*w/2,(1-v[:,1]/v[:,3])*ht/2]
def shot(config):
    for name,value in zip(JOINTS,np.r_[config,WRIST]):p.resetJointState(body,ids[name],float(value))
    im=p.getCameraImage(w,ht,view,projection,renderer=p.ER_TINY_RENDERER,shadow=0)
    return np.asarray(im[2]).reshape(ht,w,4),np.asarray(im[4]).reshape(ht,w)
initial,mask=shot(q);end,em=shot(q+tau*u)
canvas=initial.copy();select=((em&((1<<24)-1))==body)&((mask&((1<<24)-1))!=body)
canvas[select,:3]=.35*end[select,:3]+.65*canvas[select,:3]
teal='#007f89';orange='#df8b24';ink='#183f4b'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'mathtext.fontset':'stix','font.size':14})
fig=plt.figure(figsize=(10,4.4),facecolor='white')
ax=fig.add_axes([0,.10,.48,.84]);ax.imshow(canvas);ax.axis('off')
xy=project(cs);ax.plot(*xy.T,color=orange,lw=2.5)
ax.scatter(*xy[0],s=48,color=teal,edgecolors='white',zorder=5)
ax.scatter(*xy[-1],s=48,color=orange,edgecolors='white',zorder=5)
ax.annotate('Current pose',xy[0],xytext=(.04,.81),textcoords='axes fraction',color=teal,
            arrowprops=dict(arrowstyle='->',color=teal),fontsize=14)
ax.annotate('Endpoint ghost',xy[-1],xytext=(.60,.24),textcoords='axes fraction',color=orange,
            arrowprops=dict(arrowstyle='->',color=orange),fontsize=14)
occupied=np.argwhere(np.any(canvas[:,:,:3]<235,axis=2))
lo=occupied.min(0);hi=occupied.max(0);ax.set_xlim(lo[1]-45,hi[1]+45);ax.set_ylim(hi[0]+45,lo[0]-45)
ax.set_title('(b) One certified hop',color=ink,fontsize=18)

# Obstacle-normal plane, so clearance lengths and sphere cross-sections are exact.
tangent=cs[-1]-c-normal*np.dot(cs[-1]-c,normal)
tangent/=np.linalg.norm(tangent)
basis=np.array([normal,tangent]);path=(cs-c)@basis.T
distance=float(np.linalg.norm(closest-c));worst2=np.array([D,0.])
ax=fig.add_axes([.54,.20,.44,.65]);ax.set_aspect('equal');ax.axis('off')
# Draw the actual intersecting box cross-section in this plane via edge intersections.
from scipy.spatial import ConvexHull
import itertools
corners=np.array([BOX+HALF*np.array(s) for s in itertools.product([-1,1],repeat=3)])
plane_normal=np.cross(normal,tangent);points=[]
for i in range(8):
    for j in range(i+1,8):
        if np.count_nonzero(corners[i]!=corners[j])!=1:continue
        da=np.dot(corners[i]-c,plane_normal);db=np.dot(corners[j]-c,plane_normal)
        if da*db<=0 and abs(da-db)>1e-12:points.append(corners[i]+da/(da-db)*(corners[j]-corners[i]))
points=(np.array(points)-c)@basis.T;poly=points[ConvexHull(points).vertices]
ax.fill(poly[:,0],poly[:,1],color='#c9c2b3',ec='#978f80',zorder=0)
ax.add_patch(Circle((0,0),r,fc=teal,ec=teal,alpha=.8))
ax.add_patch(Circle(path[-1],r,fc=teal,ec=teal,alpha=.20))
ax.add_patch(Circle(worst2,r,fill=False,ec=orange,lw=2.2,ls='--'))
ax.plot(path[:,0],path[:,1],color=teal,lw=2)
ax.annotate('',path[-1],path[0],arrowprops=dict(arrowstyle='->',color=teal,lw=2))
ax.annotate('',worst2,(0,0),arrowprops=dict(arrowstyle='->',color=orange,lw=2))
ax.text(-r,-r-.014,'Current',color=teal,ha='center',fontsize=13)
ax.annotate('Actual endpoint',path[-1],xytext=(-48,32),textcoords='offset points',color=teal,fontsize=12,
            arrowprops=dict(arrowstyle='-',color=teal))
ax.annotate('Worst-case toward box',worst2+[0,-r],xytext=(-5,-38),textcoords='offset points',ha='center',
            color=orange,fontsize=12,arrowprops=dict(arrowstyle='-',color=orange))
ax.text(distance+.012,.015,'Actual\nobstacle',color=ink,fontsize=13)
ax.annotate('',(distance,-r-.035),(r,-r-.035),arrowprops=dict(arrowstyle='<->',color=orange))
ax.text((distance+r)/2,-r-.047,r'$h_0$ + margin',ha='center',fontsize=11,color=orange)
ax.set_xlim(-r-.065,distance+.12);ax.set_ylim(-r-.105,max(r,path[-1,1]+r)+.06)
ax.set_title('(c) Actual motion versus worst case',color=ink,fontsize=17)
fig.text(.5,.045,rf'Hold $u$ for $\tau={tau*1000:.1f}$ ms:  $\|p(\tau)-p(0)\|\leq D(\tau)=h_0$',ha='center',fontsize=15)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'bottom_row.{ext}',dpi=260)
original=Image.open('/home/ax-ml-ubu24/Downloads/Combined_MainFig.png')
combined=plt.figure(figsize=(10,10));ax=combined.add_axes([0,.42,1,.58]);ax.imshow(original);ax.set_ylim(875,0);ax.axis('off')
ax=combined.add_axes([0,0,1,.43]);ax.imshow(Image.open(OUT/'bottom_row.png'));ax.axis('off')
combined.savefig(OUT/'figure1_mockup.png',dpi=220)
(OUT/'audit.json').write_text(json.dumps(dict(q=q.tolist(),u=u.tolist(),tau=tau,sphere=sphere,
    D=D,margin=MARGIN,actual_displacement=float(np.linalg.norm(cs[-1]-c)),min_sampled_clearance=clearance,
    worst_case_center=worst.tolist(),actual_center=cs[-1].tolist(),
    scope='Python Level-2 world collision. Dashed sphere is the scalar-bound worst-case witness toward the box, not a reachable robot pose.'),indent=2)+'\n')
print('Saved',OUT,'tau',tau,'D',D,'actual',np.linalg.norm(cs[-1]-c));p.disconnect()
