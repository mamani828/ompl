"""L2-envelope helpers shared by the corrected teaser panels."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hold_time import Geometry, level2
from render_ur5_certificate_teaser import URDF, JOINTS, WRIST, BOX, HALF


def ur5_rows(model,q):
    full=np.r_[q,WRIST]
    transforms={'offset_link':np.eye(4)}
    axes={};origins={}
    for parent,child,T,index,axis in model.chain:
        frame=transforms[parent]@T
        if index>=0:
            axes[index]=frame[:3,:3]@axis
            origins[index]=frame[:3,3]
            rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(axis*full[index]).as_matrix()
            frame=frame@rot
        transforms[child]=frame
    rows=[]
    h=model.clearance(q)
    for i,((name,local,r),weights) in enumerate(zip(model.spheres,model.A)):
        active=np.flatnonzero(weights>1e-12)
        if not len(active):continue
        count=int(active[-1])+1
        center=(transforms[name]@np.r_[local,1])[:3]
        segments=[origins[k+1]-origins[k] for k in range(count-1)]+[center-origins[count-1]]
        rows.append((Geometry([axes[k] for k in range(count)],segments),float(h[i]),count))
    return rows


def ur5_l2_route(model,old):
    # Retain the existing geometric path, remove only collinear old L1 samples.
    waypoints=[old[0]]
    for i in range(1,len(old)-1):
        a=old[i]-old[i-1];b=old[i+1]-old[i]
        if np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b))>1e-7:
            waypoints.append(old[i])
    waypoints.append(old[-1])
    route=[waypoints[0]];ratios=[];records=[]
    for target in waypoints[1:]:
        for _ in range(300):
            q=route[-1];delta=target-q;distance=np.linalg.norm(delta)
            if distance<1e-8:break
            v=delta/distance;T=min(.42,distance)
            rows=ur5_rows(model,q)
            holds=[level2(g,v[:n],h,T).tau for g,h,n in rows]
            tau=min(holds)
            step=T if tau>=T-1e-12 else .88*tau
            if step<1e-7:raise RuntimeError('L2 UR5 step stalled')
            l1=min(T,float(np.min(model.clearance(q)/np.maximum(model.A@np.abs(v),1e-15))))
            ratios.append(step/l1)
            route.append(q+step*v)
            records.append({'horizon':T,'l2_limit':tau,'step':step,'l1_limit':l1})
    route=np.array(route)
    audit=min(float(model.clearance((1-t)*a+t*b).min())
              for a,b in zip(route[:-1],route[1:]) for t in np.linspace(0,1,41))
    assert audit>0
    return route,{'method':'hold_time.level2, pure L2, all moving UR5 sphere/environment rows',
                  'old_L1_hops':len(old)-1,'L2_hops':len(route)-1,
                  'largest_local_step_over_L1_limit':max(ratios),'dense_clearance_m':audit,
                  'records':records,'route':route.tolist()}


def render_plate(route,path):
    p.connect(p.DIRECT)
    robot=p.loadURDF(str(URDF),useFixedBase=True)
    ids={p.getJointInfo(robot,j)[1].decode():j for j in range(p.getNumJoints(robot))}
    for j in range(p.getNumJoints(robot)):
        name=p.getJointInfo(robot,j)[12].decode();color=[.73,.78,.81,1]
        if name in ['shoulder_link','wrist_1_link','wrist_2_link','wrist_3_link']:color=[.25,.57,.70,1]
        if 'robotiq' in name or 'fts_' in name:color=[.23,.28,.31,1]
        p.changeVisualShape(robot,j,rgbaColor=color,specularColor=[.45]*3)
    def box(center,half,color):
        s=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
        p.createMultiBody(baseMass=0,baseVisualShapeIndex=s,basePosition=center)
    box([0,0,.44],[.145,.145,.44],[.80,.83,.85,1])
    box([0,0,.895],[.19,.19,.018],[.30,.36,.40,1])
    box(BOX,HALF,[.72,.69,.63,1])
    view=p.computeViewMatrix([2.5,-3.2,2.7],[.03,.26,1.28],[0,0,1])
    projection=p.computeProjectionMatrixFOV(21,2400/1500,.05,10)
    def shot():
        result=p.getCameraImage(2400,1500,view,projection,renderer=p.ER_TINY_RENDERER,
                               shadow=1,lightDirection=[-3,-4,8],lightColor=[1,1,1],
                               lightAmbientCoeff=.65,lightDiffuseCoeff=.5,lightSpecularCoeff=.15,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return np.asarray(result[2],dtype=np.uint8).reshape(1500,2400,4),np.asarray(result[4]).reshape(1500,2400)
    def pose(q):
        for name,value in zip(JOINTS,np.r_[q,WRIST]):p.resetJointState(robot,ids[name],float(value))
    pose(route[0]);old,oldmask=shot()
    pose(route[1]);new,newmask=shot()
    p.resetBasePositionAndOrientation(robot,[0,0,-20],[0,0,0,1]);base,_=shot()
    out=base.copy();mask=(oldmask&((1<<24)-1))==robot
    out[mask,:3]=(.24*old[mask,:3]+.76*base[mask,:3]).astype(np.uint8)
    mask=(newmask&((1<<24)-1))==robot;out[mask]=new[mask]
    Image.fromarray(out).save(path);p.disconnect()


def planar_rows(q,clearance,fk):
    points=fk(q);h=clearance(q)
    first=np.r_[points[1],0];second=np.r_[points[2]-points[1],0]
    elbow=Geometry([[0,0,1]],[first])
    tip=Geometry([[0,0,1],[0,0,1]],[first,second])
    # Any material point on link 2 is an affine combination of its endpoints.
    # Bounding both endpoints bounds the full capsule centerline displacement.
    return [(elbow,min(h),1),(tip,h[1],2)]


def planar_hold(rows,v,T):
    return min(level2(g,v[:n],float(h),T).tau for g,h,n in rows)


def planar_region(q,clearance,fk,rays=360):
    rows=planar_rows(q,clearance,fk)
    angles=np.linspace(0,2*np.pi,rays,endpoint=False)
    directions=np.c_[np.cos(angles),np.sin(angles)]
    radii=np.array([planar_hold(rows,u,1.5) for u in directions])
    return q+directions*radii[:,None],directions,radii,rows
