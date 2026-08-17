#pragma once

#include <memory>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/ConfigurationOperations.h>

namespace ompl::cbf
{
    /// Real-vector storage with robot-specific topology and metric.
    template <typename Robot>
    class RobotStateSpace : public base::RealVectorStateSpace
    {
    public:
        using Configuration = typename Robot::Configuration;
        using Operations = RobotConfigurationOperations<Robot>;
        static constexpr int dimension = static_cast<int>(Robot::nJoints);

        explicit RobotStateSpace(const Configuration &speed = Robot::velocityLimits())
          : RobotStateSpace(speed, Robot::lowerBounds(), Robot::upperBounds())
        {
        }

        RobotStateSpace(const Configuration &speed, const Configuration &lower,
                        const Configuration &upper)
          : base::RealVectorStateSpace(dimension), speed_(speed)
        {
            base::RealVectorBounds bounds(dimension);
            for (int j = 0; j < dimension; ++j)
            {
                bounds.setLow(j, lower[j]);
                bounds.setHigh(j, upper[j]);
            }
            setBounds(bounds);
        }

        double distance(const base::State *state1, const base::State *state2) const override
        {
            return Operations::distance(configurationOf(state1), configurationOf(state2), speed_);
        }

        void interpolate(const base::State *from, const base::State *to, double fraction,
                         base::State *state) const override
        {
            const Configuration a = configurationOf(from);
            const Configuration b = configurationOf(to);
            setState(state, Operations::interpolate(a, b, fraction));
        }

        void enforceBounds(base::State *state) const override
        {
            setState(state, Operations::normalize(configurationOf(state)));
            base::RealVectorStateSpace::enforceBounds(state);
        }

        static Configuration configurationOf(const base::State *state)
        {
            Configuration q;
            const double *values = state->as<StateType>()->values;
            for (int j = 0; j < dimension; ++j)
                q[j] = values[j];
            return q;
        }

        static void setState(base::State *state, const Configuration &q)
        {
            double *values = state->as<StateType>()->values;
            for (int j = 0; j < dimension; ++j)
                values[j] = q[j];
        }

        const Configuration &speed() const
        {
            return speed_;
        }

    private:
        Configuration speed_;
    };
}  // namespace ompl::cbf
