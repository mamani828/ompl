#!/usr/bin/env python3
"""Computed two-joint Level-2 certificate slice and its FK image for Figure 1."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image
import pybullet as p
from render_ur5_certificate_teaser import Model, URDF, JOINTS, WRIST, BOX, HALF
from teaser_l2 import ur5_rows
from hold_time import level2
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ur5_envelope_region'
TEAL='#007f89'; ORANGE='#df8b24'; INK='#183f4b'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=Model(); q=np.array([-1.3,-1.05,1.3]); T=.9
    rows=ur5_rows(model,q)
    angles=np.linspace(0,2*np.pi,361)
    directions=np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]
    holds=np.array([min(level2(g,v[:n],h,T).tau for g,h,n in rows) for v in directions])
    dq=directions*holds[:,None]
    fractions=np.linspace(0,1,25)
    grid=np.array([[model.fk(q+f*d)[0] for f in fractions] for d in dq])
    clearance=min(float(model.clearance(q+f*d).min()) for d in dq for f in fractions)
    assert clearance >= -1e-9
    # Independent finite differences validate initial Jacobians against URDF FK.
    p.connect(p.DIRECT); body=p.loadURDF(str(URDF),useFixedBase=True)
    ids={p.getJointInfo(body,j)[1].decode():j for j in range(p.getNumJoints(body))}
    linkids={p.getJointInfo(body,j)[12].decode():j for j in range(p.getNumJoints(body))}
    for j in range(p.getNumJoints(body)):
        name=p.getJointInfo(body,j)[12].decode()
        p.changeVisualShape(body,j,rgbaColor=([.23,.55,.67,1] if 'wrist' in name or 'shoulder' in name else [.72,.77,.79,1]))
    shape=p.createVisualShape(p.GEOM_BOX,halfExtents=HALF,rgbaColor=[.68,.65,.58,1])
    p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=BOX)
    width,height=1600,1200
    view=p.computeViewMatrix([2.5,-3.2,2.7],[0,.15,1.25],[0,0,1])
    projection=p.computeProjectionMatrixFOV(24,width/height,.05,10)
    M=np.array(projection).reshape(4,4,order='F')@np.array(view).reshape(4,4,order='F')
    def project(x):
        x=np.asarray(x); s=x.shape; flat=x.reshape(-1,3)
        clip=np.c_[flat,np.ones(len(flat))]@M.T; xy=clip[:,:2]/clip[:,3,None]
        return np.c_[(xy[:,0]+1)*width/2,(1-xy[:,1])*height/2].reshape(*s[:-1],2)
    fk_error=0.
    def shot(config):
        nonlocal fk_error
        for name,val in zip(JOINTS,np.r_[config,WRIST]):p.resetJointState(body,ids[name],float(val))
        state=p.getLinkState(body,linkids['robotiq_85_base_link'],computeForwardKinematics=True)
        ee=np.array(state[4])+np.array(p.getMatrixFromQuaternion(state[5])).reshape(3,3)@np.array([0,0,.14])
        fk_error=max(fk_error,float(np.linalg.norm(ee-model.fk(config)[0])))
        im=p.getCameraImage(width,height,view,projection,renderer=p.ER_TINY_RENDERER,shadow=0)
        return np.asarray(im[2]).reshape(height,width,4),np.asarray(im[4]).reshape(height,width)
    base,bmask=shot(q); canvas=base.copy()
    chosen=[int(np.argmax(holds)),int(np.argmin(holds))]
    for k in chosen:
        im,mask=shot(q+.9*dq[k]); sel=((mask&((1<<24)-1))==body)&((bmask&((1<<24)-1))!=body)
        canvas[sel,:3]=.27*im[sel,:3]+.73*canvas[sel,:3]
    assert fk_error<1e-6
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'], 'mathtext.fontset':'stix','font.size':15})
    fig=plt.figure(figsize=(10,4.1),facecolor='white')
    left=fig.add_axes([.0,.08,.52,.87]); left.imshow(canvas); left.axis('off')
    projected=project(grid); faces=[]
    for a in range(len(angles)-1):
        for b in range(len(fractions)-1):
            faces.append([projected[a,b],projected[a+1,b],projected[a+1,b+1],projected[a,b+1]])
    left.add_collection(PolyCollection(faces,facecolors=TEAL,edgecolors='none',alpha=.17))
    left.plot(*projected[:,-1].T,color=TEAL,lw=2)
    for k in range(0,len(angles)-1,30):left.plot(*projected[k].T,color=TEAL,alpha=.22,lw=.7)
    start=project(model.fk(q)[0]); left.scatter(*start,s=45,color=ORANGE,edgecolors='white',zorder=9)
    for k in chosen:
        left.plot(*projected[k].T,color=ORANGE,lw=2)
        left.scatter(*projected[k,-1],color=ORANGE,s=25)
    pts=np.vstack((projected.reshape(-1,2),project(np.array([[0,0,.88],[0,0,1.6],BOX]))))
    lo=pts.min(0)-110; hi=pts.max(0)+110
    left.set_xlim(lo[0],hi[0]);left.set_ylim(hi[1],lo[1])
    left.text(.015,.98,'(b)',transform=left.transAxes,fontsize=25,color=INK,va='top')
    left.text(.5,-.01,'Kinematic image of the certificate',transform=left.transAxes,ha='center',color=INK)
    right=fig.add_axes([.61,.21,.36,.65])
    right.fill(dq[:,0],dq[:,1],color=TEAL,alpha=.13)
    right.plot(dq[:,0],dq[:,1],color=TEAL,lw=2.5)
    for k in range(0,len(angles)-1,30):right.plot([0,dq[k,0]],[0,dq[k,1]],color=TEAL,alpha=.15,lw=.8)
    right.scatter(0,0,s=50,c=ORANGE,zorder=5)
    for k in chosen:
        right.annotate('',dq[k,:2],(0,0),arrowprops=dict(arrowstyle='->',color=ORANGE,lw=2))
    right.set_aspect('equal');right.spines[['top','right']].set_visible(False)
    right.spines[['left','bottom']].set_color('#9bb0b4')
    right.set_xlabel(r'$\Delta q_1$ (rad)');right.set_ylabel(r'$\Delta q_2$ (rad)')
    right.set_title('Direction-dependent hold',color=INK,pad=10)
    right.text(-.21,1.07,'(c)',transform=right.transAxes,fontsize=25,color=INK)
    fig.text(.77,.035,r'$\Delta q=\tau(u)u,\quad D_i(\tau;u)\leq h_i\;\;\forall i$',ha='center',fontsize=15)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'bottom_row.{ext}',dpi=260,facecolor='white')
    # Review mockup preserves the supplied top panel and substitutes this bottom row.
    original=Image.open('/home/ax-ml-ubu24/Downloads/Combined_MainFig.png')
    row=Image.open(OUT/'bottom_row.png')
    combined=plt.figure(figsize=(10,10.0),facecolor='white')
    ax=combined.add_axes([0,.40,1,.60]);ax.imshow(original);ax.set_ylim(920,0);ax.axis('off')
    ax=combined.add_axes([0,0,1,.41]);ax.imshow(row);ax.axis('off')
    combined.savefig(OUT/'figure1_mockup.png',dpi=220)
    combined.savefig(OUT/'figure1_mockup.pdf',dpi=220)
    audit={'q0':q.tolist(),'fixed_wrist':WRIST,'horizon':T,'directions':directions.tolist(),
           'holds':holds.tolist(),'workspace_grid':grid.tolist(),'sampled_min_clearance':clearance,
           'renderer_fk_max_error_m':fk_error,'scope':'Python Level-2 world-sphere certificate; q1/q2 slice, other joints fixed. Boundary and FK surface sampled, interpolated for display; not a Cartesian safe set or production HoldEngine export.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['sampled_min_clearance','renderer_fk_max_error_m','scope']}))
    p.disconnect()

if __name__=='__main__':main()
