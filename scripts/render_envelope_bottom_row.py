#!/usr/bin/env python3
"""Figure 1 bottom-row candidate: URDF render and computed certificate geometry.

Uses the Python Level-2 certificate, not a recorded production planner hop.
World sphere obstacle only; all distances use the URDF collision spheres.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pybullet as p
from plot_ur5_kinematic_envelope_explainer import UR5, URDF, ROOT, JOINTS
from hold_time import level2, speed_capped_bound

OUT = ROOT/'results/ur5_envelope_bottom_row'
BLUE, GOLD, RED, INK = '#247caa', '#df9b20', '#c76363', '#233747'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = UR5()
    q = np.array([-1.1, -1.02, 1.28, -1.38, -.92, .48])
    u = np.array([.48, -.36, .56, .30, -.34, .24])
    tf, frames, c = robot.fk(q)
    radii = np.array([s[2] for s in robot.spheres])
    obstacle = robot.fk(q+1.2*u)[2][34]; ro = .065; T = 1.6
    h = np.linalg.norm(c-obstacle, axis=1)-radii-ro
    rows = []
    for i in range(len(c)):
        idx, g = robot.influence(i, tf, frames, c[i])
        if g is None:
            rows.append((T, 0., 0., 0.))
        else:
            r = level2(g, u[idx], h[i], T)
            rows.append((r.tau, float(np.linalg.norm(g.J0@u[idx])), r.envelopes.C, r.envelopes.V))
    k = int(np.argmin(np.array(rows)[:, 0])); tau, S, C, V = rows[k]
    ts = np.linspace(0, tau, 301)
    cc = np.array([robot.fk(q+t*u)[2] for t in ts])
    db = np.array([[speed_capped_bound(t, *row[1:]) for row in rows] for t in ts])
    residual = np.linalg.norm(cc-c, axis=2)-db
    assert residual.max() < 1e-10
    assert np.min(h-db[-1]) > -1e-10

    # Actual robot meshes rendered by a deterministic software renderer.
    p.connect(p.DIRECT)
    body = p.loadURDF(str(URDF), useFixedBase=True)
    joints = {p.getJointInfo(body, j)[1].decode(): j for j in range(p.getNumJoints(body))}
    for j in range(-1, p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode() if j >= 0 else ''
        color = [.25, .52, .65, 1] if 'wrist' in name or 'shoulder' in name else [.66, .71, .74, 1]
        p.changeVisualShape(body, j, rgbaColor=color)
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=ro, rgbaColor=[.76, .34, .34, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=obstacle)
    w, ht = 1400, 1050
    view = p.computeViewMatrix([2.0, -2.5, 2.1], [0, .22, 1.10], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(28, w/ht, .05, 10)
    def shot(t):
        for name, val in zip(JOINTS, q+t*u):
            p.resetJointState(body, joints[name], float(val))
        im = p.getCameraImage(w, ht, view, proj, renderer=p.ER_TINY_RENDERER,
                             lightDirection=[-3, -4, 8], shadow=0)
        return np.asarray(im[2]).reshape(ht, w, 4), np.asarray(im[4]).reshape(ht, w)
    current, mask = shot(0)
    canvas = current.copy()
    for t, alpha in [(tau/2, .20), (tau, .38)]:
        ghost, gm = shot(t)
        select = ((gm & ((1 << 24)-1)) == body) & ((mask & ((1 << 24)-1)) != body)
        canvas[select, :3] = alpha*ghost[select, :3]+(1-alpha)*canvas[select, :3]
    mat = np.array(proj).reshape(4, 4, order='F')@np.array(view).reshape(4, 4, order='F')
    def project(points):
        z = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))]@mat.T
        z = z[:, :2]/z[:, 3, None]
        return np.c_[(z[:, 0]+1)*w/2, (1-z[:, 1])*ht/2]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                         'axes.labelcolor': INK, 'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(15, 4.5), facecolor='white')
    ax = fig.add_axes([.0, .12, .40, .82]); ax.imshow(canvas); ax.axis('off')
    xy = project(c[k])[0]
    ax.scatter(*xy, s=110, facecolor=GOLD, edgecolor='white', lw=1.5)
    ax.annotate('Track a collision sphere', xy, xytext=(70, 130), fontsize=12,
                arrowprops=dict(arrowstyle='->', color=INK, connectionstyle='arc3,rad=-.2'))
    trace = project(cc[:, 34]); ax.plot(*trace.T, color=BLUE, lw=2)
    ax.text(.06, .03, 'Solid: initial pose   •   Ghosts: held control', transform=ax.transAxes, fontsize=10)
    ax.set_title('(a) Hold the same joint velocity', loc='left', fontsize=13, weight='bold')

    # Orthographic projection onto the plane containing initial center and obstacle.
    # Balls project to exact disks; this panel is a projection, not a planar robot.
    e1 = (obstacle-c[k])/np.linalg.norm(obstacle-c[k])
    delta = cc[-1, k]-c[k]; e2 = delta-e1*np.dot(delta, e1); e2 /= np.linalg.norm(e2)
    trace2 = (cc[:, k]-c[k])@np.array([e1, e2]).T
    distance = np.linalg.norm(obstacle-c[k]); r = radii[k]; D = db[-1, k]
    ax = fig.add_axes([.405, .22, .32, .64]); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(Circle((0, 0), r+D, fc=GOLD, ec=GOLD, alpha=.16, lw=2))
    ax.add_patch(Circle((0, 0), r+D, fill=False, ec=GOLD, lw=2))
    ax.add_patch(Circle((distance, 0), ro, fc=RED, ec=RED, alpha=.70))
    for j in [0, 150, 300]:
        ax.add_patch(Circle(trace2[j], r, fc=BLUE, ec=BLUE, alpha=.10 if j else .25, lw=1.4))
    ax.plot(*trace2.T, color=BLUE, lw=2.3)
    ax.scatter(0, 0, s=20, c=INK)
    ax.annotate('', (r+D, 0), (r, 0), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.4))
    ax.text(r+D/2, -.019, '$h_0$', ha='center')
    ax.text(-.015, -.036, '$p(0)$', ha='right')
    ax.annotate('Computed enclosing ball', (0, r+D), xytext=(-.13, r+D+.055),
                fontsize=11, arrowprops=dict(arrowstyle='-', color=GOLD))
    ax.text(distance, -ro-.029, 'Obstacle', ha='center', color=RED)
    ax.text(-r-D, -r-D-.04, r'Every moved sphere lies inside $B(p(0),r+D(t))$', fontsize=10)
    ax.set_xlim(-r-D-.03, distance+ro+.03); ax.set_ylim(-r-D-.065, r+D+.085)
    ax.set_title('(b) Bound the whole sweep', loc='left', fontsize=13, weight='bold', pad=18)

    ax = fig.add_axes([.77, .38, .21, .40])
    speed_t = np.linspace(0, tau, 301)
    ax.fill_between(speed_t, 0, np.minimum(S+C*speed_t, V), color=GOLD, alpha=.22)
    ax.plot(speed_t, np.minimum(S+C*speed_t, V), color=GOLD, lw=2.6)
    dt = 1e-5
    vel = np.array([(robot.fk(q+(t+dt)*u)[2][k]-robot.fk(q+(t-dt)*u)[2][k])/(2*dt) for t in speed_t])
    ax.plot(speed_t, np.linalg.norm(vel, axis=1), color=BLUE, lw=2)
    ax.set_xlim(0, tau); ax.set_ylim(0, max(V, S+C*tau)*1.1)
    ax.set_xticks([0, tau], ['0', r'$\tau$']); ax.set_yticks([])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Held time'); ax.set_ylabel('Sphere speed')
    ax.text(.03, .88, r'$\min(S+Ct,V)$', transform=ax.transAxes, color=GOLD)
    ax.text(.03, .10, r'Shaded area = $D(t)$', transform=ax.transAxes, fontsize=10)
    fig.text(.77, .88, '(c) Integrate the speed bound', fontsize=13, weight='bold')
    fig.text(.77, .24, r'$\|p(t)-p(0)\|\leq D(t)\leq h_0$', fontsize=13)
    fig.text(.77, .16, 'The swept sphere cannot reach the obstacle.', fontsize=10)
    fig.text(.50, .025, f'Computed UR5 example: certified hold {tau:.3f} s   |   Initial clearance {h[k]*1000:.1f} mm',
             ha='center', fontsize=11, color=INK)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/f'envelope_bottom_row.{ext}', dpi=240, facecolor='white')
    audit = dict(q=q.tolist(), u=u.tolist(), horizon=T, tau=tau, binding_sphere=k,
                 obstacle=obstacle.tolist(), obstacle_radius=ro, sphere_radius=r,
                 S=S, C=C, V=V, D=D, h0=float(h[k]),
                 max_sampled_displacement_residual=float(residual.max()),
                 note='Python Level-2, analytic sphere obstacle, world collision only; float64 evaluation.')
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (OUT/'caption.txt').write_text('Kinematic envelope certificate. (a) The UR5 holds a constant joint velocity; ghost poses are evaluated by forward kinematics. (b) Each moving collision sphere is contained in a ball of radius r+D(t) centered at its initial center (orthographic projection shown). (c) Integrating the bounded speed gives D(t). While D(t) is at most the initial clearance, the enclosing ball cannot intersect the obstacle interior. The robot hold is the minimum certificate over its collision spheres. This illustration uses the Python Level-2 world-collision certificate and an analytic sphere obstacle.\n')
    print(json.dumps(audit, indent=2)); p.disconnect()


if __name__ == '__main__':
    main()
