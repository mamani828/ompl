#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ompl::robots
{
    /// Add a world-frame planar holonomic base to a generated fixed-base sphere robot.
    /// Configuration order is [base_x, base_y, base_yaw, arm joints...].
    template <typename ArmRobot>
    class HolonomicMobileManipulator
    {
    public:
        static constexpr std::size_t nBaseJoints = 3;
        static constexpr std::size_t nJoints = nBaseJoints + ArmRobot::nJoints;
        static constexpr std::size_t nSpheres = ArmRobot::nSpheres;
        static constexpr std::size_t nSelfPairs = ArmRobot::nSelfPairs;

        using Configuration = Eigen::Matrix<double, nJoints, 1>;
        using ArmConfiguration = typename ArmRobot::Configuration;
        using Jacobian = Eigen::Matrix<double, 3, nJoints>;
        using SphereCenters = Eigen::Matrix<double, 3, nSpheres>;

        struct ConfigurationOperations
        {
            static double wrapYaw(double yaw)
            {
                constexpr double pi = 3.14159265358979323846;
                return std::remainder(yaw, 2.0 * pi);
            }

            static Configuration difference(const Configuration &from, const Configuration &to)
            {
                Configuration delta = to - from;
                delta[2] = wrapYaw(delta[2]);
                return delta;
            }

            static Configuration normalize(const Configuration &q)
            {
                Configuration out = q;
                out[2] = wrapYaw(out[2]);
                return out;
            }

            static Configuration interpolate(const Configuration &from, const Configuration &to,
                                             double fraction)
            {
                return normalize(from + fraction * difference(from, to));
            }

            static Configuration integrate(const Configuration &q, const Configuration &velocity,
                                           double duration)
            {
                return normalize(q + duration * velocity);
            }

            /// Mobile planning distance is synchronized travel time, not a mixture of
            /// metres and radians.
            static double distance(const Configuration &from, const Configuration &to,
                                   const Configuration &speed)
            {
                return duration(from, to, speed);
            }

            static double distance(const Configuration &from, const Configuration &to)
            {
                return duration(from, to, HolonomicMobileManipulator::velocityLimits());
            }

            static double duration(const Configuration &from, const Configuration &to,
                                   const Configuration &speed)
            {
                return difference(from, to).cwiseAbs().cwiseQuotient(speed).maxCoeff();
            }

            static double defaultReachTolerance(const Configuration &, double stepSize)
            {
                return stepSize;
            }
        };

        struct Kinematics
        {
            typename ArmRobot::Kinematics arm;
            Eigen::Vector3d baseOrigin;
        };

        explicit HolonomicMobileManipulator(const ArmRobot &arm = ArmRobot()) : arm_(arm)
        {
        }

        static ArmConfiguration armConfiguration(const Configuration &q)
        {
            return q.template tail<ArmRobot::nJoints>();
        }

        Eigen::Isometry3d basePose(const Configuration &q) const
        {
            Eigen::Isometry3d pose = arm_.basePose();
            pose.pretranslate(Eigen::Vector3d(q[0], q[1], 0.0));
            pose.rotate(Eigen::AngleAxisd(q[2], Eigen::Vector3d::UnitZ()));
            return pose;
        }

        Kinematics kinematics(const Configuration &q) const
        {
            const Eigen::Isometry3d pose = basePose(q);
            return {ArmRobot::kinematicsAtBase(armConfiguration(q), pose), pose.translation()};
        }

        static Eigen::Vector3d sphereCenter(const Kinematics &kin, std::size_t sphere)
        {
            return ArmRobot::sphereCenter(kin.arm, sphere);
        }

        static Jacobian sphereJacobian(const Kinematics &kin, std::size_t sphere)
        {
            const Eigen::Vector3d center = sphereCenter(kin, sphere);
            Jacobian jacobian = Jacobian::Zero();
            jacobian.col(0) = Eigen::Vector3d::UnitX();
            jacobian.col(1) = Eigen::Vector3d::UnitY();
            jacobian.col(2) = Eigen::Vector3d::UnitZ().cross(center - kin.baseOrigin);
            jacobian.template rightCols<ArmRobot::nJoints>() =
                ArmRobot::sphereJacobian(kin.arm, sphere);
            return jacobian;
        }

        static Eigen::Vector3d tipPosition(const Kinematics &kin, bool left)
        {
            return ArmRobot::tipPosition(kin.arm, left);
        }

        static Jacobian tipJacobian(const Kinematics &kin, bool left)
        {
            const Eigen::Vector3d tip = tipPosition(kin, left);
            Jacobian jacobian = Jacobian::Zero();
            jacobian.col(0) = Eigen::Vector3d::UnitX();
            jacobian.col(1) = Eigen::Vector3d::UnitY();
            jacobian.col(2) = Eigen::Vector3d::UnitZ().cross(tip - kin.baseOrigin);
            jacobian.template rightCols<ArmRobot::nJoints>() = ArmRobot::tipJacobian(kin.arm, left);
            return jacobian;
        }

        static const auto &spheres()
        {
            return ArmRobot::spheres();
        }

        static const auto &selfPairs()
        {
            return ArmRobot::selfPairs();
        }

        static const auto &steps()
        {
            return ArmRobot::steps();
        }

        static Configuration lowerBounds()
        {
            constexpr double pi = 3.14159265358979323846;
            Configuration q;
            q << -1.0, -0.65, -pi, ArmRobot::lowerBounds();
            return q;
        }

        static Configuration upperBounds()
        {
            constexpr double pi = 3.14159265358979323846;
            Configuration q;
            q << 0.35, 0.65, pi, ArmRobot::upperBounds();
            return q;
        }

        static Configuration velocityLimits()
        {
            Configuration speed;
            speed.template head<3>() << 0.35, 0.35, 0.6;
            speed.template tail<ArmRobot::nJoints>() =
                ArmRobot::velocityLimits().cwiseMin(ArmConfiguration::Constant(1.2));
            return speed;
        }

        const ArmRobot &armRobot() const
        {
            return arm_;
        }

    private:
        ArmRobot arm_;
    };
}  // namespace ompl::robots
