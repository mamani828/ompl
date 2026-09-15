#!/usr/bin/env python3
"""UR5 adaptive hops above a matched planar workspace / C-space illustration."""
from pathlib import Path
import json
import shutil
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.spatial import HalfspaceIntersection, ConvexHull
import pybullet as p
from render_ur5_certificate_teaser import Model, plan
from teaser_l2 import ur5_l2_route, render_plate, planar_rows, planar_hold, planar_region

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/adaptive_l2_teaser'
SOURCE=ROOT/'results/ur5_certificate_teaser'
TEAL='#087f87'; INK='#254754'; MUTED='#81969d'; GOLD='#d98d2f'
L=np.array([.9,.7]); R=.045
# One disc, 1.32 from the base. The distance is the whole design: link 1 is 0.9 long,
# so with the obstacle any nearer the *elbow* cannot clear it and the configuration space
# is cut into disconnected halves -- measured, at 1.16 there is no free q2 at all for
# q1 in [0.4, 0.6]. Out here only link 2 can reach it, so the obstacle becomes an island
# with free space on both sides, which is what lets a plan curve around rather than stop.
OBSTACLES=np.array([[1.32*np.cos(.45),1.32*np.sin(.45)]])
RADII=np.array([.24])


def planar_fk(q):
    q=np.asarray(q)
    angles=np.cumsum(q,axis=-1)
    vectors=L[:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    zero=np.zeros(q.shape[:-1]+(1,2))
    return np.concatenate([zero,np.cumsum(vectors,axis=-2)],axis=-2)


def clearance(q):
    joints=planar_fk(q)
    start=joints[...,:-1,:];end=joints[...,1:,:]
    delta=end-start
    # Exact circle-to-capsule separation, one clearance per link.
    offset=OBSTACLES-start[...,None,:]
    t=np.clip(np.sum(offset*delta[...,None,:],axis=-1)/np.sum(delta*delta,axis=-1)[...,None],0,1)
    nearest=start[...,None,:]+t[...,None]*delta[...,None,:]
    return np.min(np.linalg.norm(nearest-OBSTACLES,axis=-1)-RADII-R,axis=-1)


def integrate(a,b):
    points=[a.copy()]
    for _ in range(100):
        delta=b-points[-1]
        if np.linalg.norm(delta)<1e-9:break
        distance=np.linalg.norm(delta)
        rows=planar_rows(points[-1],clearance,planar_fk)
        tau=planar_hold(rows,delta/distance,distance)
        fraction=1. if tau>=distance-1e-12 else .86*tau/distance
        if fraction*np.linalg.norm(delta)<1e-6:raise RuntimeError('Planar path stalled')
        points.append(points[-1]+fraction*delta)
    return np.array(points)


def ur5_panel(ax,model,edges,route,standalone=False):
    ax.imshow(Image.open(OUT/'simulator_l2.png'))
    view=p.computeViewMatrix([2.5,-3.2,2.7],[.03,.26,1.28],[0,0,1])
    proj=p.computeProjectionMatrixFOV(21,2400/1500,.05,10)
    VP=np.array(proj).reshape(4,4,order='F')@np.array(view).reshape(4,4,order='F')
    def project(points):
        points=np.asarray(points);h=np.c_[points,np.ones(len(points))]@VP.T
        h=h[:,:3]/h[:,3,None]
        return np.c_[(h[:,0]+1)*1200,(1-h[:,1])*750]
    for edge in edges:
        pts=[]
        for a,b in zip(edge[:-1],edge[1:]):
            pts.extend(model.fk((1-t)*a+t*b)[0] for t in np.linspace(0,1,5))
        xy=project(pts)
        ax.plot(*xy.T,lw=.6,color='#8fa6af',alpha=.44,zorder=2)
    trace=np.array([model.fk(q)[0] for q in route]);xy=project(trace)
    dq=np.linalg.norm(np.diff(route,axis=0),axis=1)
    small=np.r_[False,dq<np.quantile(dq,.38)]
    for a,b in zip(route[:-1],route[1:]):
        line=project([model.fk((1-t)*a+t*b)[0] for t in np.linspace(0,1,15)])
        ax.plot(*line.T,color=TEAL,lw=1.6,zorder=3)
    ax.scatter(*xy[~small].T,s=20,c=TEAL,edgecolors='white',linewidths=.65,zorder=4)
    ax.scatter(*xy[small].T,s=20,c=GOLD,edgecolors='white',linewidths=.55,zorder=5)
    ax.scatter(*xy[:2].T,s=60,c=TEAL,edgecolors='white',linewidths=1,zorder=6)
    ax.annotate('',xy[1],xytext=xy[0],arrowprops={'arrowstyle':'-|>','color':TEAL,'lw':1.6,'shrinkA':5,'shrinkB':5},zorder=6)
    # No callouts: the gold/teal dots carry the adaptive step on their own, and what
    # the colours mean belongs in the caption rather than inside the plate.
    ax.set_xlim(550,2220);ax.set_ylim(1260,260);ax.axis('off')
    return {'joint_step_min_rad':float(dq.min()),'joint_step_max_rad':float(dq.max()),
            'small_step_color_threshold_rad':float(np.quantile(dq,.38))}


def arm(ax,q,alpha=1):
    pts=planar_fk(q)
    # Exact capsule silhouettes in workspace units, matching the collision map.
    for a,b in zip(pts[:-1],pts[1:]):
        tangent=(b-a)/np.linalg.norm(b-a)
        normal=np.array([-tangent[1],tangent[0]])
        for radius,color,zorder in [(R,'#c3cfd3',4),(.70*R,'#7daebc',5)]:
            n=radius*normal
            ax.add_patch(Polygon([a+n,b+n,b-n,a-n],facecolor=color,edgecolor='none',alpha=alpha,zorder=zorder))
            for center in [a,b]:ax.add_patch(Circle(center,radius,facecolor=color,edgecolor='none',alpha=alpha,zorder=zorder))
    ax.scatter(*pts.T,s=[110,90,38],c=[INK,TEAL,TEAL],edgecolors='white',linewidths=1,alpha=alpha,zorder=6)


# Both ends comfortably clear, and the straight line between them passes through the
# obstacle, so a plan has to go around it rather than merely slow down.
START=np.array([1.52,.93]); GOAL=np.array([-.57,.31])


def clearance_gradient(q,eps=1e-4):
    """Central difference of the minimum clearance. Two joints, so four evaluations."""
    g=np.empty(2)
    for i in range(2):
        step=np.zeros(2);step[i]=eps
        g[i]=(clearance(q+step).min()-clearance(q-step).min())/(2*eps)
    return g


def admissible_direction(u,g,h,kappa,previous=None):
    """Closest *unit* direction to `u` with `grad h . u >= -kappa h`, or None.

    Projecting u onto the half-plane and renormalising does not work: the projection
    lands on the boundary with `g . u' = -kappa h`, and scaling it back to unit length
    scales that product up by `1/|u'| > 1`, which violates the condition again. So solve
    on the circle directly -- with `ghat = g/|g|` and `c = -kappa h/|g|`, the admissible
    set is the arc `ghat . u >= c`, whose boundary is `c ghat +- sqrt(1-c^2) ghat_perp`.
    Take whichever boundary vector is closer to the nominal direction.
    """
    norm=np.linalg.norm(g)
    if norm<1e-12:return u
    ghat=g/norm;c=-kappa*h/norm
    if ghat@u>=c:return u
    if c>=1.:return None
    perp=np.array([-ghat[1],ghat[0]]);sin=np.sqrt(max(0.,1.-c*c))
    # Two boundary solutions, one either side of the gradient. Choosing purely by
    # closeness to the nominal direction lets the pick flip from one to the other as the
    # nominal rotates, which chatters the edge into a zigzag. Bias toward whichever
    # continues the previous heading, so the rollout slides along an obstacle instead.
    reference=u if previous is None else u+.9*previous
    return max([c*ghat+sin*perp,c*ghat-sin*perp],key=lambda w:w@reference)


def cbf_extend(q,target,kappa=1.,horizon=.9,max_steps=30,utilisation=.86,floor=.02):
    """One tree extension, integrated under the barrier condition.

    Both mechanisms the figure is about happen here. The *direction* is the
    straight-line one, deflected onto the admissible cone whenever heading at the
    obstacle would spend clearance too fast -- so an edge curves along the obstacle
    instead of stopping at it. The *step* is the second-order hold along the accepted
    direction, so endpoints crowd together near the obstacle and spread out in the open.
    """
    points=[np.asarray(q,float)];heading=None
    for _ in range(max_steps):
        delta=target-points[-1];remaining=np.linalg.norm(delta)
        if remaining<floor:break
        h=clearance(points[-1]).min()
        if h<=0:break
        u=admissible_direction(delta/remaining,clearance_gradient(points[-1]),h,kappa,
                               heading)
        if u is None:break
        heading=u
        rows=planar_rows(points[-1],clearance,planar_fk)
        advance=min(utilisation*planar_hold(rows,u,min(remaining,horizon)),remaining)
        # Below the floor the rollout is grinding along the surface rather than
        # travelling; stop and let the planner branch from here instead. Without this it
        # spends its whole step budget covering a few hundredths of a radian.
        if advance<floor:break
        points.append(points[-1]+advance*u)
    return np.array(points)


def cbf_rrt(start,goal,seed=11,iterations=420,reach=.85,goal_bias=.14,tolerance=.14,
            margin=.55):
    """RRT whose every extension is a barrier-filtered rollout.

    Sampling bounds are derived from the endpoints rather than fixed, with `margin` of
    room on each side so the tree can route around an obstacle that sits between them.
    Hard-coding them silently starves the search the moment the scene changes.
    """
    rng=np.random.default_rng(seed)
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    lo=np.minimum(start,goal)-margin;hi=np.maximum(start,goal)+margin
    nodes=[np.asarray(start,float)];parents=[-1];edges=[];reached=None
    for _ in range(iterations):
        sample=goal if rng.random()<goal_bias else lo+(hi-lo)*rng.random(2)
        i=int(np.argmin(np.linalg.norm(np.array(nodes)-sample,axis=1)))
        delta=sample-nodes[i];span=np.linalg.norm(delta)
        if span<1e-9:continue
        target=nodes[i]+delta/span*min(span,reach)
        trajectory=cbf_extend(nodes[i],target)
        if len(trajectory)<2:continue
        nodes.append(trajectory[-1]);parents.append(i);edges.append((i,len(nodes)-1,trajectory))
        if np.linalg.norm(trajectory[-1]-goal)<tolerance:
            reached=len(nodes)-1;break
    return nodes,parents,edges,reached


def shortcut(path,seed=0,rounds=400,tolerance=.02,max_steps=140):
    """Remove the zigzag an RRT leaves behind, without leaving the filtered world.

    Each attempt is itself a barrier-filtered rollout between two waypoints, so a
    shortcut that lands on its target is admissible by construction -- there is no
    straight-line segment spliced in that the filter never saw. A candidate is kept only
    if it actually arrives *and* is shorter in joint-space length than the run it
    replaces, so the path length is monotonically non-increasing.
    """
    length=lambda p:float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    rng=np.random.default_rng(seed)
    path=[np.asarray(q,float) for q in path]
    for _ in range(rounds):
        if len(path)<4:break
        i=int(rng.integers(0,len(path)-2));j=int(rng.integers(i+2,len(path)))
        segment=cbf_extend(path[i],path[j],max_steps=max_steps)
        if np.linalg.norm(segment[-1]-path[j])>tolerance:continue
        if length(segment)>=length(np.array(path[i:j+1])):continue
        path[i:j+1]=[np.asarray(q,float) for q in segment]
    return np.array(path)


def planar_panels(work,config):
    # A single RRT seed either connects or does not; try a few rather than pinning the
    # figure to whichever one happened to work, which silently breaks on any scene edit.
    for seed in (3,5,11,17,23):
        nodes,parents,edges,reached=cbf_rrt(START,GOAL,seed=seed,iterations=900)
        if reached is not None:break
    else:
        raise RuntimeError('no seed connected START to GOAL; check the scene')
    by_child={child:traj for _,child,traj in edges}
    # The solution, walked back from the node that reached the goal.
    chain=[];node=reached
    while node is not None and node>0:
        chain.append(by_child[node]);node=parents[node]
    chain.reverse()
    # Consecutive edges share their joining node, so drop the repeat -- otherwise every
    # junction contributes a zero-length step and the hop statistics are meaningless.
    raw=np.vstack([chain[0]]+[c[1:] for c in chain[1:]]) if chain else np.array([START])
    # Best of a few restarts: one shortcut pass is a random walk over pair choices and
    # its quality varies, so keep the shortest rather than whatever the first seed gave.
    length=lambda p:float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    solution=min((shortcut(raw,seed=k,rounds=900) for k in range(4)),key=length)
    # A shortcut whose far end is the last waypoint lands within `tolerance` of it rather
    # than on it, and that slack accumulates over rounds -- it walked the endpoint 0.39
    # rad off the goal. Close the remainder with one more filtered rollout, on a finer
    # floor so it can actually arrive.
    closing=cbf_extend(solution[-1],GOAL,max_steps=80,floor=.004)
    if np.linalg.norm(closing[-1]-GOAL)<np.linalg.norm(solution[-1]-GOAL):
        solution=np.vstack([solution,closing[1:]])
    goal_q=solution[-1]

    # ---- (b) workspace: the arm at both ends, and where the tip travelled.
    for center,radius in zip(OBSTACLES,RADII):
        work.add_patch(Circle(center,radius,facecolor='#c7bfb0',edgecolor='#9f9688',lw=.8,zorder=2))
    work.add_patch(Polygon([[-.12,-.1],[.12,-.1],[.085,-.01],[-.085,-.01]],closed=True,color='#677b84',zorder=2))
    # The motion itself: poses along the plan, fading back in time, with the tip trace
    # threading them. Tree branches are left to (c) -- as workspace traces they read as
    # scribble and bury the arm, which is the subject here.
    # Candidate plans as tip traces, faint; the executed one solid. Only two arm poses
    # are drawn -- start and goal -- because every pose shares the base joint, so a fan
    # of silhouettes piles up there no matter how it is faded.
    # Ranked by joint-space length, not by point count: an edge that crawled along the
    # obstacle carries many waypoints over very little ground, so `len` picked those and
    # dropped the branches that actually go somewhere.
    reach=lambda t:float(np.linalg.norm(np.diff(t,axis=0),axis=1).sum())
    for trajectory in sorted((t for _,_,t in edges),key=reach,reverse=True)[:6]:
        work.plot(*planar_fk(trajectory)[:,-1,:].T,color=TEAL,lw=.9,alpha=.26,zorder=3)
    work.plot(*planar_fk(solution)[:,-1,:].T,color=TEAL,lw=2.,zorder=4)
    arm(work,START,.30);arm(work,goal_q)
    origin=planar_fk(START)[-1]
    work.annotate(bold('start'),origin,xytext=(-14,-24),textcoords='offset points',ha='center',
                  fontsize=FS['label'],color=GOLD,zorder=9,**BOLD)
    work.scatter(*origin,s=70,c=GOLD,edgecolors='white',lw=.9,zorder=9)
    target=planar_fk(GOAL)[-1]
    work.scatter(*target,s=170,marker='*',c=GOLD,edgecolors='white',lw=.8,zorder=9)
    work.annotate(bold('goal'),target,xytext=(14,-6),textcoords='offset points',
                  fontsize=FS['label'],color=GOLD,zorder=9,**BOLD)
    # Cropped to what is actually drawn -- arm poses, tip trace and the obstacle -- not
    # to the arm's whole reach square, which left most of the panel empty.
    drawn=np.vstack([planar_fk(solution).reshape(-1,2),
                     OBSTACLES-RADII[:,None],OBSTACLES+RADII[:,None],
                     np.array([[-.16,-.14]])])
    lo=drawn.min(0)-.12;hi=drawn.max(0)+.12
    centre=.5*(lo+hi);half=.5*max(hi-lo)
    work.set_xlim(centre[0]-half,centre[0]+half);work.set_ylim(centre[1]-half,centre[1]+half)
    work.set_aspect('equal');work.axis('off')
    work.set_title(bold('(b)'),loc='left',fontsize=FS['title'],color=INK,pad=8,**BOLD)

    # ---- (c) configuration space: the same tree, bent around the obstacle.
    pad=.10
    every=np.vstack([t for _,_,t in edges]+[np.array([START,GOAL])])
    xlo,ylo=every.min(0)-pad;xhi,yhi=every.max(0)+pad
    gx=np.linspace(xlo,xhi,520);gy=np.linspace(ylo,yhi,520)
    xx,yy=np.meshgrid(gx,gy)
    field=np.min(clearance(np.stack([xx,yy],axis=-1)),axis=-1)
    config.contourf(xx,yy,field,levels=[-10,0,10],colors=['#ddd7ce','#ffffff'])
    config.contour(xx,yy,field,levels=[0],colors=['#b2a99a'],linewidths=.9)
    for _,_,trajectory in edges:
        config.plot(*trajectory.T,color='#7fb0b4',lw=.8,alpha=.75,zorder=3)

    config.plot(*solution.T,color=TEAL,lw=1.7,zorder=5)
    # One marker per integration endpoint, all the same size, coloured by how much room
    # the certificate had: gold where clearance is tightest and the hops come close
    # together, teal once there is space and they stretch out. Same gold/teal reading as
    # panel (a), and the threshold is a quantile of the path's own clearance so it
    # follows the scene rather than being a tuned constant.
    room=np.array([clearance(q).min() for q in solution])
    near=room<=np.quantile(room,.38)
    # A lone gold node among teal ones is classification flicker -- the threshold is a
    # quantile, so a single waypoint whose clearance dips just under it reads as a stray
    # dot rather than as "here the certificate was tight". Keep a node gold only if a
    # neighbour on the path is gold too, so only genuine runs are marked.
    adjacent=np.zeros_like(near)
    adjacent[:-1]|=near[1:];adjacent[1:]|=near[:-1]
    near&=adjacent
    config.scatter(*solution[~near].T,s=17,c='white',edgecolors=TEAL,linewidths=1.,zorder=6)
    config.scatter(*solution[near].T,s=17,c='white',edgecolors=GOLD,linewidths=1.2,zorder=7)
    config.scatter(*START,s=52,c=GOLD,edgecolors='white',lw=.8,zorder=7)
    config.scatter(*GOAL,s=150,marker='*',c=GOLD,edgecolors='white',lw=.8,zorder=7)
    config.annotate(bold('start'),START,xytext=(11,-17),textcoords='offset points',
                    fontsize=FS['label'],color=GOLD,zorder=8,**BOLD)
    config.annotate(bold('goal'),GOAL,xytext=(13,-4),textcoords='offset points',
                    fontsize=FS['label'],color=GOLD,zorder=8,**BOLD)
    config.set_xlim(xlo,xhi);config.set_ylim(ylo,yhi)
    # Bold in the maths too. `\\mathbf` is applied to the symbol rather than the whole
    # group so the subscript is not dragged upright with it, and the unit is bolded
    # through `bold()` so the label is one weight on either rendering path.
    axis=lambda i: rf'$\mathbf{{q}}_{i}$ '+bold('(rad)')
    config.set_xlabel(axis(1),fontsize=FS['label'],color=INK,labelpad=4,**BOLD)
    config.set_ylabel(axis(2),fontsize=FS['label'],color=INK,labelpad=4,**BOLD)
    config.set_title(bold('(c)'),loc='left',fontsize=FS['title'],color=INK,pad=8,**BOLD)
    config.tick_params(labelsize=FS['tick'],colors=MUTED,length=3)
    for spine in ['top','right']:config.spines[spine].set_visible(False)
    for spine in ['left','bottom']:config.spines[spine].set_color('#bdc9cd')
    config.text(.025,.035,bold('Collision'),transform=config.transAxes,color='#958976',
                fontsize=FS['sub'],**BOLD)

    # Every state on every edge must be clear -- the filter is the only thing keeping
    # them so, since nothing here calls a collision checker.
    dense=np.vstack([a+(b-a)*t for _,_,trajectory in edges
                     for a,b in zip(trajectory[:-1],trajectory[1:])
                     for t in np.linspace(0,1,25)[:,None]])
    worst=float(clearance(dense).min());assert worst>0,worst
    dense_solution=np.vstack([a+(b-a)*t for a,b in zip(solution[:-1],solution[1:])
                              for t in np.linspace(0,1,41)[:,None]])
    solution_clearance=float(clearance(dense_solution).min());assert solution_clearance>0
    steps=np.linalg.norm(np.diff(solution,axis=0),axis=1)
    chord=float(np.linalg.norm(solution[-1]-solution[0]))
    length=lambda p:float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    return {'start':START.tolist(),'goal':GOAL.tolist(),
            'waypoints_before_shortcut':int(len(raw)),
            'length_before_shortcut_rad':length(raw),'length_after_shortcut_rad':length(solution),
            'minimum_clearance_along_solution_m':solution_clearance,
            'solution_chord_rad':chord,
            'method':'RRT whose extensions are barrier-filtered rollouts: direction '
                     'deflected onto the admissible cone, step length from hold_time.level2',
            'straight_line_start_to_goal_blocked':True,
            'tree_nodes':len(nodes),'tree_edges':len(edges),
            'solution_waypoints':int(len(solution)),
            'solution_step_min_rad':float(steps.min()),'solution_step_max_rad':float(steps.max()),
            'solution_step_ratio':float(steps.max()/steps.min()),
            'minimum_clearance_over_all_edges_m':worst,
            'goal_gap_rad':float(np.linalg.norm(goal_q-GOAL))}


FS = {'title': 20, 'annot': 19, 'label': 17, 'sub': 17, 'tick': 14}

# Bold has to be said twice, because the two rendering paths take it differently.
# `fontweight` works for mathtext but is *ignored* under `text.usetex`, where matplotlib
# hands the string to LaTeX unwrapped -- which silently dropped every bold label the
# first time TeX was switched on. `\\textbf` covers that path and is a no-op on the other.
BOLD = {'fontweight': 'bold'}


def bold(text):
    """Mark `text` bold on whichever path is active. See BOLD."""
    return rf'\textbf{{{text}}}' if plt.rcParams['text.usetex'] else text


def latex_fonts():
    """Set the figure in Liberation Serif, the metric-compatible clone of Times New Roman.

    Times New Roman itself is proprietary and not installed; `fc-match` on this machine
    already substitutes Liberation Serif for it. Being a *metric* clone matters -- the
    glyph widths match, so figure text sits at the same weight and width as body text in
    a Times-set paper rather than merely resembling it.

    Maths comes from STIX, because none of the Times clones ships an italic maths face.
    STIX is Times-like, so `$q_1$` pairs with the surrounding text, but text and maths do
    come from two families; only Computer Modern and STIX supply both from one design.

    Unlike Computer Modern this face has a real bold and a real U+2212, so no `cmb10`
    naming trick and no mathtext-formatter workaround for negative ticks.
    """
    common={'font.family':'serif','pdf.fonttype':42,'ps.fonttype':42}
    if shutil.which('latex') and shutil.which('dvipng'):
        # A real TeX run, with `mathptmx`: Times for the text *and* the maths, from one
        # package. That is the one thing the fallback below cannot do -- no Times clone
        # ships an italic maths face, so it has to borrow STIX for `$q_1$`.
        plt.rcParams.update({**common,'text.usetex':True,
                             'font.serif':['Times','Nimbus Roman'],
                             'axes.unicode_minus':False,
                             'text.latex.preamble':r'\usepackage{mathptmx}'})
        return 'usetex + mathptmx (Times text and maths)'
    plt.rcParams.update({**common,'text.usetex':False,
                         'font.serif':['Liberation Serif','Nimbus Roman','DejaVu Serif'],
                         'mathtext.fontset':'stix',
                         'axes.unicode_minus':True,
                         'axes.formatter.use_mathtext':False})
    return 'Liberation Serif (Times metric clone) + STIX maths'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print('fonts:', latex_fonts(), flush=True)
    model=Model();edges,old_route=plan(model)
    route,ur5_audit=ur5_l2_route(model,old_route)
    print('UR5 resampled:',ur5_audit['old_L1_hops'],'L1 hops ->',ur5_audit['L2_hops'],'L2 hops',flush=True)
    render_plate(route,OUT/'simulator_l2.png')
    report={'ur5_l2_audit':ur5_audit}
    fig=plt.figure(figsize=(10,9.1),facecolor='white')
    top=fig.add_axes([.005,.415,.99,.585])
    report['ur5']=ur5_panel(top,model,edges,route)
    top.text(.030,.985,bold('(a)'),transform=top.transAxes,
             fontsize=FS['title'],color=INK,va='top',**BOLD)
    work=fig.add_axes([.015,.045,.40,.345])
    config=fig.add_axes([.545,.105,.40,.295])
    report['planar']=planar_panels(work,config)
    fig.savefig(OUT/'combined.png',dpi=300,bbox_inches='tight',pad_inches=.02)
    fig.savefig(OUT/'combined.pdf',dpi=300,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));fig.subplots_adjust(0,0,1,1)
    ur5_panel(ax,model,edges,route,standalone=True)
    fig.savefig(OUT/'ur5_adaptive.png',dpi=300,bbox_inches='tight',pad_inches=.02)
    fig.savefig(OUT/'ur5_adaptive.pdf',dpi=300,bbox_inches='tight',pad_inches=.02);plt.close(fig)
    fig=plt.figure(figsize=(10,3.8))
    work=fig.add_axes([.04,.10,.42,.8]);config=fig.add_axes([.57,.15,.38,.73])
    planar_panels(work,config)
    fig.savefig(OUT/'planar_cspace.png',dpi=300,bbox_inches='tight',pad_inches=.02)
    fig.savefig(OUT/'planar_cspace.pdf',dpi=300,bbox_inches='tight',pad_inches=.02);plt.close(fig)

    # Each planar panel on its own, for a layout that places them apart. `planar_panels`
    # draws into both axes it is given, so the panel not being kept goes to a scratch
    # figure that is discarded -- cheaper than splitting the function and risking the two
    # variants drifting apart.
    for name,keep in (('planar_workspace',0),('planar_config',1)):
        fig=plt.figure(figsize=(5,4.2))
        ax=fig.add_axes([.13,.13,.83,.78] if keep else [.06,.10,.90,.82])
        scratch=plt.figure()
        other=scratch.add_axes([0,0,1,1])
        planar_panels(*( (other,ax) if keep else (ax,other) ))
        fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.02)
        fig.savefig(OUT/f'{name}.pdf',dpi=300,bbox_inches='tight',pad_inches=.02)
        plt.close(fig);plt.close(scratch)
    report['scope']='Pure hold_time.level2 in both panels. UR5: existing RRT geometric route resampled with L2 sphere/environment holds. Planar: exact capsule-circle geometry, capsule endpoints bounded by L2 envelopes.'
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'ur5_hops':ur5_audit['L2_hops'], 'planar':{k:v for k,v in report['planar'].items() if k not in ['l2_region','l1_reference']}},indent=2))


if __name__=='__main__':main()
