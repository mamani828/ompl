#!/usr/bin/env python3
"""Reproducible PyBullet teaser; illustrative planar CBF planning experiment.

No learned/generated imagery. Dots are actual integration states. This is a
point-robot illustration, not a projection of a recorded UR5 benchmark tree.
"""
from pathlib import Path
import json
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results/adaptive_teaser'
OBS = [(np.array([0., 0.]), .77), (np.array([1.8, -1.5]), .42)]
START = np.array([-3.3, -.25])
GOAL = np.array([3.25, .55])
KAPPA = 3.
MARGIN = .09


def rollout(start, target, horizon=2.5):
    q = start.copy()
    points, times = [q.copy()], []
    elapsed = 0.
    while elapsed < horizon and np.linalg.norm(target-q) > .045:
        v = target-q
        v /= np.linalg.norm(v)
        hs = np.array([np.linalg.norm(q-c)-r-MARGIN for c, r in OBS])
        # Euclidean projection onto planar CBF halfspaces. Obstacles are
        # separated; iterate to satisfy any simultaneously active rows.
        normals = [(q-c)/np.linalg.norm(q-c) for c, _ in OBS]
        for _ in range(12):
            for n, h in zip(normals, hs):
                deficit = -KAPPA*h - n@v
                if deficit > 0:
                    v += deficit*n
        speed = np.linalg.norm(v)
        if speed < .008:
            break
        # Distance is 1-Lipschitz. tau <= h/||v|| - 1/kappa
        # certifies a no-op interval; active filtering uses a short base step.
        noop = np.min(hs)/speed - 1/KAPPA
        dt = min(.32, max(.045, .90*noop), horizon-elapsed, np.linalg.norm(target-q)/speed)
        nxt = q + dt*v
        assert all(np.linalg.norm(nxt-c) > r+MARGIN-1e-9 for c,r in OBS)
        points.append(nxt.copy())
        times.append(dt)
        elapsed += dt
        q = nxt
    return np.asarray(points), times


def tree():
    rng = np.random.default_rng(24)
    nodes = [START.copy()]
    edges = []
    # Grow a reproducible exploratory tree, retaining each integrated edge.
    for _ in range(72):
        target = rng.uniform([-3.5,-1.85],[3.45,1.95])
        if any(np.linalg.norm(target-c) < r+.16 for c,r in OBS):
            continue
        i = np.argmin([np.linalg.norm(q-target) for q in nodes])
        pts, ts = rollout(nodes[i], target, .85)
        if len(pts)>1 and np.linalg.norm(pts[-1]-pts[0])>.12:
            edges.append((pts,ts))
            nodes.append(pts[-1])
    # A selected tree branch, biased through an upper waypoint to avoid the
    # equilibrium of a head-on nominal command against a round obstacle.
    route, durations = [START], []
    for target in [np.array([1.4,.85]), GOAL]:
        pts, ts = rollout(route[-1], target, 12.)
        route.extend(pts[1:])
        durations.extend(ts)
    return edges, np.array(route), durations


def body(kind, xyz, color, **kwargs):
    shape = p.createVisualShape(kind, rgbaColor=color, **kwargs)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=xyz)


def segment(a,b,radius,color,z=.025):
    a,b = np.r_[a,z],np.r_[b,z]
    d = b-a
    length = np.linalg.norm(d)
    if length < 1e-8:
        return
    axis = np.cross([0,0,1],d/length)
    quat = p.getQuaternionFromAxisAngle(axis, np.pi/2)
    shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,length=length,rgbaColor=color)
    p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=(a+b)/2,baseOrientation=quat)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    edges, route, durations = tree()
    p.connect(p.DIRECT)
    body(p.GEOM_BOX,[0,0,-.09],[.97,.975,.98,1],halfExtents=[200,200,.07])
    for c,r in OBS:
        body(p.GEOM_CYLINDER,[*c,.15],[.60,.65,.70,1],radius=r,length=.30)
    muted = [.61,.72,.76,1]
    teal = [.015,.40,.43,1]
    for pts, _ in edges:
        for a,b in zip(pts[:-1],pts[1:]):
            segment(a,b,.008,muted)
        for q in pts:
            body(p.GEOM_SPHERE,[*q,.029],muted,radius=.016)
    for a,b in zip(route[:-1],route[1:]):
        segment(a,b,.016,teal,z=.045)
    for q in route:
        body(p.GEOM_SPHERE,[*q,.061],teal,radius=.032)
    for q in [START,route[-1]]:
        body(p.GEOM_CYLINDER,[*q,.066],[1,.57,.22,1],radius=.085,length=.024)
    width,height = 2400,1320
    view = p.computeViewMatrix([0,-3.6,10.8],[0,0,0],[0,1,0])
    proj = p.computeProjectionMatrix(-4.05,4.05,-2.23,2.23,10,30)
    frame = p.getCameraImage(width,height,view,proj,shadow=1,lightDirection=[-3,-4,9],
                             renderer=p.ER_TINY_RENDERER)[2]
    frame = np.asarray(frame,dtype=np.uint8).reshape(height,width,4)
    plt.imsave(OUT/'simulator.png',frame)
    # Labels are vector overlays on the actual simulator screenshot.
    def screen(q,z=0):
        v = np.array(view).reshape(4,4,order='F')
        m = np.array(proj).reshape(4,4,order='F')
        x = m@v@np.array([*q,z,1.])
        x /= x[3]
        return [(x[0]+1)*width/2,(1-x[1])*height/2]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42})
    fig,ax = plt.subplots(figsize=(12,6.6))
    fig.subplots_adjust(0,0,1,1)
    ax.imshow(frame)
    ax.axis('off')
    for q,label,offset in [(START,'Start',(-45,40)),(route[-1],'Goal',(15,-35))]:
        xy = screen(q)
        ax.annotate(label,xy,xytext=offset,textcoords='offset points',fontsize=14,weight='medium',color='#253746')
    ax.annotate('Long certified steps',screen(route[3]),xytext=(310,180),
                fontsize=17,color='#00686f',arrowprops={'arrowstyle':'-','color':'#00686f','lw':1.1})
    near = np.argmin([np.linalg.norm(q) for q in route])
    ax.annotate('Short steps near obstacles',screen(route[near]),xytext=(1160,175),
                fontsize=17,color='#00686f',arrowprops={'arrowstyle':'-','color':'#00686f','lw':1.1})
    ax.text(90,1225,'CBF-filtered tree edges',fontsize=15,color='#516e78')
    ax.text(90,1285,'Dots mark integration timesteps',fontsize=12,color='#516e78')
    fig.savefig(OUT/'teaser.png',dpi=240)
    fig.savefig(OUT/'teaser.pdf',dpi=300)
    plt.close(fig)
    p.disconnect()
    # Dense segment audit independent of the endpoint checks above.
    minimum = min(np.linalg.norm((a+(b-a)*t)-c)-r-MARGIN
                  for pts in [route]+[e[0] for e in edges]
                  for a,b in zip(pts[:-1],pts[1:])
                  for t in np.linspace(0,1,31) for c,r in OBS)
    report = {'scene':'illustrative planar point robot, not UR5 benchmark data',
              'tree_edges':len(edges),'selected_branch_steps':len(durations),
              'minimum_dt':min(durations),'maximum_dt':max(durations),
              'dense_audit_minimum_clearance_beyond_margin':minimum,
              'goal_error':float(np.linalg.norm(route[-1]-GOAL)),
              'route':route.tolist(),'durations':durations}
    (OUT/'rollout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['route','durations']},indent=2))


if __name__ == '__main__':
    main()
