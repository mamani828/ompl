#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace ompl::cbf
{
    /// Euclidean configuration arithmetic used by every existing fixed-base robot.
    template <typename Configuration>
    struct EuclideanConfigurationOperations
    {
        static Configuration difference(const Configuration &from, const Configuration &to)
        {
            return to - from;
        }

        static Configuration interpolate(const Configuration &from, const Configuration &to,
                                         double fraction)
        {
            return from + fraction * difference(from, to);
        }

        static Configuration integrate(const Configuration &q, const Configuration &velocity,
                                       double duration)
        {
            return q + duration * velocity;
        }

        static Configuration normalize(const Configuration &q)
        {
            return q;
        }

        static double distance(const Configuration &from, const Configuration &to,
                               const Configuration & /*speed*/)
        {
            return difference(from, to).norm();
        }

        static double distance(const Configuration &from, const Configuration &to)
        {
            return difference(from, to).norm();
        }

        static double duration(const Configuration &from, const Configuration &to,
                               const Configuration &speed)
        {
            return difference(from, to).cwiseAbs().cwiseQuotient(speed).maxCoeff();
        }

        static double defaultReachTolerance(const Configuration &speed, double stepSize)
        {
            return speed.norm() * stepSize;
        }
    };

    namespace detail
    {
        template <typename Robot, typename = void>
        struct RobotConfigurationOperationsImpl
          : EuclideanConfigurationOperations<typename Robot::Configuration>
        {
        };

        template <typename Robot>
        struct RobotConfigurationOperationsImpl<Robot, std::void_t<typename Robot::ConfigurationOperations>>
          : Robot::ConfigurationOperations
        {
        };
    }  // namespace detail

    /// Select a robot's custom configuration topology when it provides one.
    template <typename Robot>
    using RobotConfigurationOperations = detail::RobotConfigurationOperationsImpl<Robot>;
}  // namespace ompl::cbf
