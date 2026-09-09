#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Geometry>

namespace ompl::robots::detail
{
    // An enclosing cylinder or ball. Every state encloses the complete distal
    // rotational sweep, not samples or a linearization. Ball construction may
    // place its center on the current axis or at the joint origin.
    struct SweepBound
    {
        Eigen::Vector3d center;
        Eigen::Vector3d axis;
        double radius;
        double halfHeight;
        bool cylinder;
    };

    inline double perpendicular(const Eigen::Vector3d &p, const Eigen::Vector3d &axis)
    {
        return (p - p.dot(axis) * axis).norm();
    }

    // Branch over the cylinder, axial-ball and origin-ball enclosures used by
    // safe_bubble_cover/envs/lipschitz_3d.py. Keep all paths: the smallest radius
    // at this joint need not produce the smallest bound at an upstream joint.
    inline std::vector<SweepBound> sweepBounds(const std::vector<SweepBound> &states,
                                              const Eigen::Matrix3d &rotation,
                                              const Eigen::Vector3d &translation,
                                              const Eigen::Vector3d &axis)
    {
        std::vector<SweepBound> next;
        next.reserve(3 * states.size());
        for (const auto &state : states)
        {
            const Eigen::Vector3d c = rotation * state.center + translation;
            const Eigen::Vector3d a = rotation * state.axis;
            const Eigen::Vector3d axial = c.dot(axis) * axis;
            const Eigen::Vector3d p = c - axial;
            double radial, height, axialBall, originBall;
            if (state.cylinder)
            {
                const double r = state.radius, h = state.halfHeight;
                const Eigen::Vector3d projectedAxis = a - a.dot(axis) * axis;
                const double sine = projectedAxis.norm();
                const double cosine = std::abs(a.dot(axis));
                // Bound the squared projected norm, including the axial/rim
                // cross term. Omitting that term is unsound for tilted axes.
                radial = std::sqrt(std::max(0.0, p.squaredNorm() + r * r + h * h * sine * sine +
                    2 * r * p.cross(a).norm() + 2 * h * std::abs(p.dot(projectedAxis)) +
                    2 * h * r * cosine * sine));
                height = r * sine + h * cosine;
                axialBall = std::min(p.norm() + std::hypot(r, h), std::hypot(radial, height));
                originBall = std::hypot(std::abs(c.dot(a)) + h, perpendicular(c, a) + r);
            }
            else
            {
                radial = p.norm() + state.radius;
                height = state.radius;
                axialBall = radial;
                originBall = c.norm() + state.radius;
            }
            // Small outward padding protects the analytic enclosure against
            // accumulated floating-point roundoff (units: metres).
            constexpr double pad = 1e-12;
            next.push_back({axial, axis, radial + pad, height + pad, true});
            next.push_back({axial, axis, axialBall + pad, 0.0, false});
            next.push_back({Eigen::Vector3d::Zero(), axis, originBall + pad, 0.0, false});
        }
        return next;
    }
}
