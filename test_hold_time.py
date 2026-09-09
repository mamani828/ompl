"""Run with python -m unittest -v test_hold_time (requires NumPy).

Dense trajectory checks are regression tests, not the mathematical proof.
"""
import math
import unittest
import numpy as np

from hold_time import (
    Geometry, SelfCollisionPair, weighted_l1, level2, tightened_level2,
    experimental_anchored, speed_capped_time, speed_capped_bound,
    _anchored_prefix_bound, relative_self_collision_geometry,
    normal_anchored_time, self_collision_hold, minimum_self_collision_hold,
)


def rotation(z, angle):
    x, y, w = z
    K = np.array([[0, -w, y], [w, 0, -x], [-y, x, 0]])
    return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)


def along_ray(g, v, t):
    R = np.eye(3)
    axes, segments = [], []
    for z, s, rate in zip(g.axes, g.segments, v):
        axes.append(R@z)
        R = R@rotation(z, rate*t)
        segments.append(R@s)
    return Geometry(axes, segments)


class HoldTests(unittest.TestCase):
    def test_scalar_edges_and_branches(self):
        self.assertEqual(speed_capped_time(1, 0, 2, 0), math.inf)
        self.assertEqual(speed_capped_time(1, 0, 0, 1), math.inf)
        self.assertEqual(speed_capped_time(1, 2, 0, 3), .5)
        self.assertEqual(speed_capped_time(0, 0, 0, 0), 0)
        for S, C, V in [(0, 2, 3), (1, 2, 3), (3, 2, 3), (1, 0, 3)]:
            for d in [1e-12, .1, 1, 100]:
                t = speed_capped_time(d, S, C, V)
                self.assertAlmostEqual(speed_capped_bound(t, S, C, V)/d,
                                       1, places=12)
        with self.assertRaises(ValueError):
            speed_capped_time(1, 2, 1, 1)
        with self.assertRaises(ValueError):
            speed_capped_time(-1, 0, 0, 0)

    def test_fixed_spatial_pair_and_cancellation(self):
        g = Geometry([[0, 0, 1], [0, 1, 0]],
                     [[1, 0, 0], [.2, .3, .4]])
        e = g.envelopes([2, -3], 100)
        np.testing.assert_allclose(e.omega, [2, math.sqrt(13)])
        np.testing.assert_allclose(e.angular_acceleration, [0, 6])
        c = Geometry([[0, 0, 1], [0, 0, 1]],
                     [[0, 0, 0], [1, 0, 0]])
        e = c.envelopes([1, -1], 100)
        self.assertEqual(e.C, 0)
        self.assertEqual(e.V, 0)

    def test_random_spatial_envelopes(self):
        rng = np.random.default_rng(721)
        for _ in range(15):
            z = rng.normal(size=(6, 3))
            z /= np.linalg.norm(z, axis=1)[:, None]
            g = Geometry(z, rng.normal(size=(6, 3))*.2)
            v = rng.uniform(-2, 2, 6)
            T, d = .7, .12
            e = g.envelopes(v, T)
            prev = g.envelopes(v, T/2)
            self.assertGreaterEqual(e.C+1e-10, prev.C)
            self.assertGreaterEqual(e.V+1e-10, prev.V)
            for j in range(6):
                q = sum(v[:j+1]**2)
                for k in range(j+1):
                    for m in range(k+1, j+1):
                        c = (e.cosine_upper[k, m] if v[k]*v[m] >= 0
                             else e.cosine_lower[k, m])
                        q += 2*v[k]*v[m]*c
                self.assertAlmostEqual(e.omega[j]**2, q, places=10)
            results = [level2(g, v, d, T), tightened_level2(g, v, d, T, 2),
                       tightened_level2(g, v, d, T, 8),
                       experimental_anchored(g, v, d, T)]
            self.assertGreaterEqual(results[1].tau, results[0].tau)
            self.assertGreaterEqual(results[2].tau, results[1].tau)
            self.assertGreaterEqual(results[0].tau+1e-12,
                                    min(T, weighted_l1(d, v, g.reach_radii)))
            for t in np.linspace(0, T, 41):
                gt = along_ray(g, v, t)
                u, a, w, alpha = gt.initial_motion(v)
                self.assertTrue(np.all(np.linalg.norm(w, axis=1)
                                       <= e.omega+1e-10))
                self.assertTrue(np.all(np.linalg.norm(alpha, axis=1)
                                       <= e.angular_acceleration+1e-10))
                self.assertLessEqual(np.linalg.norm(u), e.V+1e-10)
                self.assertLessEqual(np.linalg.norm(a), e.C+1e-10)
                cos = gt.axes@gt.axes.T
                self.assertTrue(np.all(cos >= e.cosine_lower-1e-10))
                self.assertTrue(np.all(cos <= e.cosine_upper+1e-10))
                wp = np.vstack((np.zeros(3), w[:-1]))
                ap = np.vstack((np.zeros(3), alpha[:-1]))
                beta = np.cumsum(v[:, None]*(np.cross(ap, gt.axes) +
                    np.cross(wp, np.cross(wp, gt.axes))), axis=0)
                s = gt.segments
                jerk = np.sum(np.cross(beta, s) +
                    2*np.cross(alpha, np.cross(w, s)) +
                    np.cross(w, np.cross(alpha, s)) +
                    np.cross(w, np.cross(w, np.cross(w, s))), axis=0)
                self.assertLessEqual(np.linalg.norm(jerk), e.H+1e-10)
            for r in results:
                self.assertLessEqual(r.tau, r.horizon)
                for t in np.linspace(0, r.tau, 21):
                    delta = (along_ray(g, v, t).segments.sum(axis=0)
                             - g.segments.sum(axis=0))
                    self.assertLessEqual(np.linalg.norm(delta), d+1e-10)

    def test_prefix_and_horizon(self):
        u, a = np.array([1., 0, 0]), np.array([-1., 0, 0])
        self.assertAlmostEqual(_anchored_prefix_bound(2, u, a, 0), .5)
        g = Geometry([[0, 0, 1]], [[1, 0, 0]])
        r = level2(g, [1], 10, .1)
        self.assertGreater(r.root, r.horizon)
        self.assertEqual(r.tau, .1)
        self.assertEqual(level2(g, [0], 1, 2).tau, 2)
        self.assertEqual(level2(g, [1], 0, 2).tau, 0)
        self.assertEqual(level2(g, [1], 1, 0).tau, 0)
        with self.assertRaises(ValueError):
            g.envelopes([1], float('inf'))
        with self.assertRaises(ValueError):
            Geometry([[0, 0, 2]], [[1, 0, 0]])


class SelfCollisionTests(unittest.TestCase):
    def assert_pair_safe(self, geometry, v, point_a, radius_sum, hold):
        for t in np.linspace(0, hold, 401):
            point_b = along_ray(geometry, v, t).segments.sum(axis=0)
            self.assertGreaterEqual(np.linalg.norm(point_a-point_b)+1e-11,
                                    radius_sum)

    def test_relative_geometry_and_dense_certificate(self):
        axes = np.array([[0., 0, 1], [0, 1., 0], [1., 0, 0]])
        origins = np.array([[0., 0, 0], [.4, 0, 0], [.7, .2, 0]])
        point_a = np.array([.1, .25, 0])
        point_b = np.array([.9, .25, .2])
        g, normal = relative_self_collision_geometry(
            axes, origins, point_a, point_b, 1, 3)
        v = np.array([1.1, -.8])
        radius_sum = .35
        d0 = np.linalg.norm(point_a-point_b)-radius_sum
        T = 1.2
        hold = self_collision_hold(g, v, d0, T, normal)
        self.assertGreaterEqual(hold, min(T, weighted_l1(
            d0, v, g.reach_radii)))
        # Express A in the relative chain's frame; its offset from origin[1]
        # is fixed after common upstream motion is cancelled.
        relative_a = point_a-origins[1]
        self.assert_pair_safe(g, v, relative_a, radius_sum, hold)

    def test_approaching_tangential_and_receding(self):
        g = Geometry([[0, 0, 1]], [[1, 0, 0]])
        v, T = np.array([1.]), 1.0
        for point_a in (np.array([0., 1.4, 0]),
                        np.array([1.4, 0., 0]),
                        np.array([0., -1.4, 0])):
            radius_sum = .3
            d0 = np.linalg.norm(point_a-g.segments.sum(axis=0))-radius_sum
            normal = (point_a-g.segments.sum(axis=0))/(d0+radius_sum)
            hold = self_collision_hold(g, v, d0, T, normal)
            self.assert_pair_safe(g, v, point_a, radius_sum, hold)

    def test_same_frame_and_pair_minimum(self):
        axes = [[0, 0, 1], [0, 1, 0]]
        origins = [[0, 0, 0], [1, 0, 0]]
        g0, n0 = relative_self_collision_geometry(
            axes, origins, [0, 0, 0], [1, 0, 0], 1, 1)
        self.assertIsNone(g0)
        self.assertEqual(self_collision_hold(g0, [], .5, 2, n0), 2)
        g = Geometry([[0, 0, 1]], [[1, 0, 0]])
        p1 = SelfCollisionPair(g, np.array([1.]), .2,
                               np.array([0., 1., 0]))
        p2 = SelfCollisionPair(g, np.array([1.]), .5,
                               np.array([0., 1., 0]))
        h1 = self_collision_hold(g, p1.v, p1.d0, 2, p1.normal)
        h2 = self_collision_hold(g, p2.v, p2.d0, 2, p2.normal)
        self.assertEqual(minimum_self_collision_hold([p1, p2], 2),
                         min(h1, h2))

    def test_old_l1_is_never_lost(self):
        g = Geometry([[0, 0, 1]], [[1, 0, 0]])
        supplied = .73
        hold = self_collision_hold(g, [2], .01, 1, [0, 1, 0],
                                   l1_time=supplied)
        self.assertGreaterEqual(hold, supplied)
        self.assertLessEqual(hold, 1)


if __name__ == '__main__':
    unittest.main()
