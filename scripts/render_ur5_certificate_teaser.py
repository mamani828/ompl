#!/usr/bin/env python3
"""UR5 screenshot + exact-FK certificate slice and recorded certified RRT hops.

The illustrative planner varies the first three joints with the wrist fixed.
Certificates cover the URDF collision spheres against the displayed box, using
global serial-chain reach bounds. They are not mesh or self-collision proofs.
The drawn surface is an FK-mapped polytope boundary, not a Cartesian safe set.
"""
from pathlib import Path
import itertools
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation
import pybullet as p
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/ur5_certificate_teaser'
URDF = ROOT/'external/vamp/resources/ur5/ur5_spherized.urdf'
JOINTS = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
          'wrist_1_joint','wrist_2_joint','wrist_3_joint']
WRIST = [-1.9,-1.57,0.]
BOX = np.array([-.03,.67,1.13])
HALF = np.array([.105,.105,.31])
MARGIN = .015


def xyz(element):
    return np.fromstring(element.get('xyz','0 0 0'),sep=' ') if element is not None else np.zeros(3)


class Model:
    def __init__(self):
        root = ET.parse(URDF).getroot()
        pending = list(root.findall('joint'))
        self.chain = []
        ancestors = {'offset_link':[]}
        while pending:
            for j in pending[:]:
                parent,child = j.find('parent').get('link'),j.find('child').get('link')
                if parent not in ancestors:
                    continue
                origin = j.find('origin')
                T = np.eye(4)
                T[:3,3] = xyz(origin)
                T[:3,:3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix() if origin is not None else np.eye(3)
                index = JOINTS.index(j.get('name')) if j.get('name') in JOINTS else -1
                self.chain.append((parent,child,T,index,xyz(j.find('axis'))))
                ancestors[child] = ancestors[parent]+[(index,np.linalg.norm(T[:3,3]))]
                pending.remove(j)
        self.spheres=[]
        weights=[]
        for link in root.findall('link'):
            for col in link.findall('collision'):
                sphere = col.find('geometry/sphere')
                if sphere is None:
                    continue
                local = xyz(col.find('origin'))
                self.spheres.append((link.get('name'),local,float(sphere.get('radius'))))
                row=np.zeros(3)
                chain=ancestors[link.get('name')]
                for k,(idx,_) in enumerate(chain):
                    if 0<=idx<3:
                        row[idx]=sum(x[1] for x in chain[k+1:])+np.linalg.norm(local)
                weights.append(row)
        self.A=np.array(weights)
        self.radii=np.array([s[2] for s in self.spheres])

    def fk(self,q):
        full=np.r_[q,WRIST]
        transforms={'offset_link':np.eye(4)}
        for parent,child,T,index,axis in self.chain:
            local=T
            if index>=0:
                rot=np.eye(4)
                rot[:3,:3]=Rotation.from_rotvec(axis*full[index]).as_matrix()
                local=T@rot
            transforms[child]=transforms[parent]@local
        centers=np.array([(transforms[name]@np.r_[local,1])[:3] for name,local,r in self.spheres])
        # Tool center between the gripper fingers, fixed in the gripper base.
        ee=(transforms['robotiq_85_base_link']@np.array([0,0,.14,1]))[:3]
        return ee,centers

    def clearance(self,q):
        _,centers=self.fk(q)
        d=np.abs(centers-BOX)-HALF
        sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
        return sdf-self.radii-MARGIN

    def integrate(self,a,b):
        q=a.copy()
        points=[q.copy()]
        for _ in range(150):
            delta=b-q
            if np.linalg.norm(delta)<1e-7:
                return np.array(points)
            h=self.clearance(q)
            if h.min()<=0:
                return None
            expenditure=self.A@np.abs(delta)
            ratio=np.min(h/np.maximum(expenditure,1e-15))
            fraction=min(1.,.88*ratio,.42/max(np.linalg.norm(delta),1e-15))
            if fraction*np.linalg.norm(delta)<.001:
                return None
            q=q+fraction*delta
            points.append(q.copy())
        return None


def plan(model):
    rng=np.random.default_rng(18)
    start=np.array([-1.3,-1.05,1.3])
    goal=np.array([.5,-1.05,1.3])
    assert min(model.clearance(start))>0 and min(model.clearance(goal))>0
    nodes=[start]; parents=[-1]; edges=[]
    solved=None
    for iteration in range(2400):
        target=goal if iteration%5==0 else rng.uniform([-1.6,-1.9,.65],[.8,-.55,1.85])
        nearest=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)))
        delta=target-nodes[nearest]
        target=nodes[nearest]+delta*min(1,.36/max(np.linalg.norm(delta),1e-12))
        edge=model.integrate(nodes[nearest],target)
        if edge is None:
            continue
        nodes.append(target); parents.append(nearest); edges.append(edge)
        if np.linalg.norm(target-goal)<.40:
            edge=model.integrate(target,goal)
            if edge is not None:
                nodes.append(goal);parents.append(len(nodes)-2);edges.append(edge)
                solved=len(nodes)-1
                break
    if solved is None:
        raise RuntimeError('No route found')
    chain=[]
    while solved>=0:
        chain.append(nodes[solved]);solved=parents[solved]
    chain=chain[::-1]
    # Deterministic shortcutting, each candidate validated by the same certificate.
    shortcut=[chain[0]]
    i=0
    while i<len(chain)-1:
        for j in range(len(chain)-1,i,-1):
            if model.integrate(chain[i],chain[j]) is not None:
                shortcut.append(chain[j]);i=j;break
    route=[shortcut[0]]
    for a,b in zip(shortcut[:-1],shortcut[1:]):
        route.extend(model.integrate(a,b)[1:])
    return edges,np.array(route)


def region(model,q):
    h=model.clearance(q)
    rows=[]
    for sign in itertools.product([-1,1],repeat=3):
        for a,b in zip(model.A,h):
            if np.linalg.norm(a)>1e-10:
                rows.append(np.r_[a*sign,-b])
    # Limit the depicted slice to a local chart, still a certified subset.
    for k in range(3):
        for s in [-1,1]:
            row=np.zeros(4);row[k]=s;row[3]=-.48;rows.append(row)
    vertices=HalfspaceIntersection(np.array(rows),np.zeros(3)).intersections
    hull=ConvexHull(vertices)
    faces=[];ridges=set()
    for triangle in hull.simplices:
        v=vertices[triangle]
        # Facets are sampled through nonlinear FK, not treated as Cartesian hulls.
        n=5
        for i in range(n):
            for j in range(n-i):
                def at(a,b):
                    return model.fk(q+v[0]*(1-(a+b)/n)+v[1]*a/n+v[2]*b/n)[0]
                faces.append([at(i,j),at(i+1,j),at(i,j+1)])
                if i+j<n-1:
                    faces.append([at(i+1,j),at(i+1,j+1),at(i,j+1)])
        for a,b in itertools.combinations(triangle,2):
            ridges.add(tuple(sorted((int(a),int(b)))))
    lines=[]
    for a,b in ridges:
        # Suppress diagonals between coplanar triangles of the original polytope.
        incident=[k for k,t in enumerate(hull.simplices) if a in t and b in t]
        if len(incident)==2 and np.linalg.norm(hull.equations[incident[0]]-hull.equations[incident[1]])<1e-6:
            continue
        lines.append(np.array([model.fk(q+(1-t)*vertices[a]+t*vertices[b])[0] for t in np.linspace(0,1,18)]))
    return np.array(faces),lines,vertices


def render(model,edges,route):
    p.connect(p.DIRECT)
    robot=p.loadURDF(str(URDF),useFixedBase=True)
    for link in range(p.getNumJoints(robot)):
        name=p.getJointInfo(robot,link)[12].decode()
        color=[.73,.78,.81,1]
        if name in ['shoulder_link','wrist_1_link','wrist_2_link','wrist_3_link']:
            color=[.25,.57,.70,1]
        if 'robotiq' in name or 'fts_' in name:
            color=[.23,.28,.31,1]
        p.changeVisualShape(robot,link,rgbaColor=color,specularColor=[.45,.45,.45])
    joints={p.getJointInfo(robot,j)[1].decode():j for j in range(p.getNumJoints(robot))}
    def pose(q):
        for name,value in zip(JOINTS,np.r_[q,WRIST]):p.resetJointState(robot,joints[name],float(value))
    def box(center,half,color):
        s=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
        return p.createMultiBody(baseMass=0,baseVisualShapeIndex=s,basePosition=center)
    box([0,0,.44],[.145,.145,.44],[.80,.83,.85,1])
    box([0,0,.895],[.19,.19,.018],[.30,.36,.40,1])
    box(BOX,HALF,[.72,.69,.63,1])
    # Floor kept below crop: the figure is a clean studio-style simulator view.
    trace=np.array([model.fk(q)[0] for q in route])
    lengths=np.linalg.norm(np.diff(trace,axis=0),axis=1)
    chosen=int(np.argmax(lengths))
    q0,q1=route[chosen:chosen+2]
    gripper=next(j for j in range(p.getNumJoints(robot)) if p.getJointInfo(robot,j)[12].decode()=='robotiq_85_base_link')
    fk_error=0.
    for q in [q0,q1,route[-1]]:
        pose(q)
        state=p.getLinkState(robot,gripper,computeForwardKinematics=True)
        bullet_ee=np.array(state[4])+np.array(p.getMatrixFromQuaternion(state[5])).reshape(3,3)@np.array([0,0,.14])
        fk_error=max(fk_error,float(np.linalg.norm(bullet_ee-model.fk(q)[0])))
    assert fk_error<1e-6
    face,ridges,vertices=region(model,q0)
    # Every depicted hop endpoint is inside the originating joint certificate.
    utilization=[float(np.max((model.A@np.abs(b-a))/model.clearance(a))) for a,b in zip(route[:-1],route[1:])]
    assert max(utilization)<=1+1e-9
    width,height=2400,1500
    cameras=[([2.5,-3.2,2.7],[.03,.26,1.28]),([2.8,2.9,2.6],[0,.25,1.30]),([2.8,-1.6,2.5],[0,.24,1.30])]
    for ci,(eye,target) in enumerate(cameras):
        view=p.computeViewMatrix(eye,target,[0,0,1])
        projection=p.computeProjectionMatrixFOV(21,width/height,.05,10)
        def shot():
            result=p.getCameraImage(width,height,view,projection,renderer=p.ER_TINY_RENDERER,
                                    shadow=1,lightDirection=[-3,-4,8],lightColor=[1,1,1],
                                    lightAmbientCoeff=.65,lightDiffuseCoeff=.5,lightSpecularCoeff=.15,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            return np.asarray(result[2],dtype=np.uint8).reshape(height,width,4),np.asarray(result[4]).reshape(height,width)
        pose(q0);old,oldmask=shot()
        pose(q1);new,newmask=shot()
        # Hide robot by moving its base, leaving a clean background plate.
        p.resetBasePositionAndOrientation(robot,[0,0,-20],[0,0,0,1]);base,_=shot()
        p.resetBasePositionAndOrientation(robot,[0,0,0],[0,0,0,1])
        composed=base.copy()
        mask=(oldmask&((1<<24)-1))==robot
        composed[mask,:3]=(.24*old[mask,:3]+.76*base[mask,:3]).astype(np.uint8)
        mask=(newmask&((1<<24)-1))==robot
        composed[mask]=new[mask]
        Image.fromarray(composed).save(OUT/f'simulator_{ci}.png')
        V=np.array(view).reshape(4,4,order='F');P=np.array(projection).reshape(4,4,order='F')
        def project(points):
            pts=np.asarray(points); flat=pts.reshape(-1,3)
            homogeneous=np.c_[flat,np.ones(len(flat))]
            clip=homogeneous@(P@V).T;clip=clip[:,:3]/clip[:,3,None]
            return np.c_[(clip[:,0]+1)*width/2,(1-clip[:,1])*height/2].reshape(*pts.shape[:-1],2)
        fig,ax=plt.subplots(figsize=(12,7.5));fig.subplots_adjust(0,0,1,1)
        ax.imshow(composed);ax.axis('off')
        # Edges use dense FK along their joint-space segments, never straight EE chords.
        for edge in edges:
            dense=[]
            for a,b in zip(edge[:-1],edge[1:]):
                dense.extend(model.fk((1-t)*a+t*b)[0] for t in np.linspace(0,1,5))
            xy=project(dense)
            ax.plot(xy[:,0],xy[:,1],color='#698c98',alpha=.48,lw=.75,zorder=2)
        order=np.argsort(np.mean((np.c_[face.reshape(-1,3),np.ones(face.size//3)]@V.T)[:,2].reshape(-1,3),axis=1))
        ax.add_collection(PolyCollection(project(face[order]),facecolors='#36b7b0',edgecolors='none',alpha=.055,zorder=3))
        ax.add_collection(LineCollection([project(line) for line in ridges],colors='#258d92',linewidths=.7,alpha=.52,zorder=4))
        for a,b in zip(route[:-1],route[1:]):
            xy=project([model.fk((1-t)*a+t*b)[0] for t in np.linspace(0,1,14)])
            ax.plot(xy[:,0],xy[:,1],color='#087b83',lw=1.8,zorder=5)
        xy=project(trace)
        ax.scatter(xy[:,0],xy[:,1],s=14,c='#087b83',edgecolors='white',linewidths=.4,zorder=6)
        for index in [chosen,chosen+1]:
            ax.scatter(*xy[index],s=75,c='#e79b3b',edgecolors='white',linewidths=1.1,zorder=7)
        a,b=xy[chosen],xy[chosen+1]
        ax.annotate('',b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#d58b2a','lw':2,'shrinkA':5,'shrinkB':5},zorder=8)
        ax.text(110,105,'UR5  /  certified adaptive integration',fontsize=15,color='#233d48')
        ax.annotate('Certified region',project(model.fk(q0)[0]),xytext=(1820,1100),
                    fontsize=13,color='#167f85',arrowprops={'arrowstyle':'-','color':'#167f85','lw':.8},zorder=9)
        ax.text(1820,1160,'FK-mapped joint-space slice',fontsize=9,color='#647f89')
        ax.text(110,1280,'Current pose',fontsize=12,color='#83949a')
        ax.text(110,1340,'Next certified pose',fontsize=12,color='#233d48')
        ax.text(110,1410,'Dots mark end-effector hops',fontsize=10,color='#647f89')
        ax.set_xlim(0,width);ax.set_ylim(height,0)
        fig.savefig(OUT/f'teaser_{ci}.png',dpi=240)
        fig.savefig(OUT/f'teaser_{ci}.pdf',dpi=300)
        if ci==0:
            # Paper composition: remove slide-style headings, crop into the
            # mechanism, and label the two FK endpoints directly.
            for artist in list(ax.texts):
                if artist.get_text():
                    artist.remove()
            ax.set_xlim(550,2200);ax.set_ylim(1430,270)
            fig.set_size_inches(10,7)
            ax.annotate('Current pose',xy[chosen],xytext=(1890,1160),
                        color='#81969c',fontsize=11,
                        arrowprops={'arrowstyle':'-','color':'#9bb0b5','lw':.75},zorder=10)
            ax.annotate('Next pose',xy[chosen+1],xytext=(1890,905),
                        color='#244e5d',fontsize=11,
                        arrowprops={'arrowstyle':'-','color':'#537884','lw':.75},zorder=10)
            boundary=project(face.reshape(-1,3))
            right=boundary[np.argmax(boundary[:,0])]
            ax.annotate('Certified region',right,xytext=(1850,1305),
                        color='#167f85',fontsize=12,
                        arrowprops={'arrowstyle':'-','color':'#36969b','lw':.8},zorder=10)
            ax.text(1850,1350,'mapped through FK',fontsize=9,color='#647f89')
            ax.annotate('End-effector path',xy[len(xy)//2],xytext=(670,380),
                        color='#167f85',fontsize=12,
                        arrowprops={'arrowstyle':'-','color':'#36969b','lw':.8},zorder=10)
            ax.text(645,1305,'Dots: integration endpoints',fontsize=10,color='#466c78')
            ax.text(645,1360,'Faint branches: RRT exploration',fontsize=9,color='#82969e')
            fig.savefig(OUT/'teaser.png',dpi=300)
            fig.savefig(OUT/'teaser.pdf',dpi=300)
            for artist in list(ax.texts):artist.remove()
            fig.savefig(OUT/'teaser_clean.png',dpi=300)
            fig.savefig(OUT/'teaser_clean.pdf',dpi=300)
        plt.close(fig)
    p.disconnect()
    return {'selected_hop':chosen,'hop_distance_m':float(lengths[chosen]),'hops':len(route)-1,
            'tree_edges':len(edges),'max_certificate_utilization':max(utilization),'fk_agreement_m':fk_error,
            'certificate_vertices':vertices.tolist(),'q0':q0.tolist(),'q1':q1.tolist(),
            'route':route.tolist(),'ee_endpoints':trace.tolist()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    model=Model()
    edges,route=plan(model)
    print('planned',len(edges),'edges;',len(route)-1,'hops',flush=True)
    report=render(model,edges,route)
    # Audit dense states independently of the certificate algebra.
    report['minimum_environment_clearance_beyond_margin']=min(float(model.clearance((1-t)*a+t*b).min())
        for a,b in zip(route[:-1],route[1:]) for t in np.linspace(0,1,41))
    report['scope']='Illustrative 3-joint UR5 RRT; fixed wrist; sphere-model environment certificate; no self-collision certificate.'
    (OUT/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['route','ee_endpoints','certificate_vertices']})


if __name__=='__main__':main()
