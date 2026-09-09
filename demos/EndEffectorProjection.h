// demos/EndEffectorProjection.h
//
// The workspace projection the KPIECE rows are discretized on: the tool-frame position
// of the robot, in metres.
//
// KPIECE needs a projection, and if none is registered OMPL invents one -- a random
// linear map of the joint angles (`ProjectionEvaluator::computeCoordinates`'s default
// path). That would be a poor thing to build a comparison on. The cells KPIECE grows
// from are supposed to stand for "parts of the problem the tree has not covered yet",
// and in a joint-space random projection they stand for nothing in particular, so the
// row would measure the projection's luck as much as the planner. It would also be
// seeded per run, adding a variance source the other rows do not have.
//
// The end effector is the natural choice instead: the obstacles are in the workspace,
// the MotionBenchMaker goals are workspace reaching tasks, and two configurations whose
// tool is in the same place really are near-substitutes for the purpose of getting
// through a shelf. Three dimensions is also what KPIECE's grid is comfortable with.
//
// The same evaluator is used by both KPIECE rows -- the collision-checked one over
// `RealVectorStateSpace` and the filtered one over `FilteredStateSpace` -- which is the
// point. `FilteredStateSpace` derives from `RealVectorStateSpace` and stores the same
// raw joint values in the same layout, so one class reads both, and the two rows are
// discretized identically. Anything left between them is the steering, not the grid.

#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

namespace ompl::cbf
{
    /// Projects a joint configuration onto the workspace position of one of \p Robot's
    /// spheres -- by default the last one, which is the far end of the kinematic chain.
    template <typename Robot>
    class EndEffectorProjection : public base::ProjectionEvaluator
    {
    public:
        using Configuration = typename Robot::Configuration;

        /// \param cellSize grid resolution in metres. The default is deliberately of the
        ///        order of the robot's own link radii: finer and every extension lands in
        ///        a fresh cell, so the cell scores KPIECE selects on carry no history;
        ///        coarser and the whole reachable workspace collapses into a few cells.
        EndEffectorProjection(const base::StateSpace *space, const Robot &robot,
                              std::size_t sphere = Robot::nSpheres - 1, double cellSize = 0.05)
          : base::ProjectionEvaluator(space), robot_(robot), sphere_(sphere), cellSize_(cellSize)
        {
        }

        unsigned int getDimension() const override
        {
            return 3;
        }

        void defaultCellSizes() override
        {
            cellSizes_.assign(3, cellSize_);
        }

        void project(const base::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override
        {
            const auto *values = state->as<base::RealVectorStateSpace::StateType>()->values;
            Configuration q;
            for (Eigen::Index j = 0; j < q.rows(); ++j)
                q[j] = values[j];
            projection = Robot::sphereCenter(robot_.kinematics(q), sphere_);
        }

    private:
        const Robot &robot_;
        std::size_t sphere_;
        double cellSize_;
    };
}  // namespace ompl::cbf
