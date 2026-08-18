// Coupled 17-DoF Reachy2 mobile-manipulation planning benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <qpmad/solver.h>

#include <ompl/base/PlannerData.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/cbf/ControlFilter.h>
#include <ompl/cbf/ExecutedPath.h>
#include <ompl/cbf/FilteredMotionValidator.h>
#include <ompl/cbf/FilteredStateSpace.h>
#include <ompl/cbf/RobotStateSpace.h>
#include <ompl/cbf/RopeShortcut.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/robots/HolonomicMobileManipulator.h>
#include <ompl/robots/Reachy2.h>
#include <ompl/sdf/GridSDF.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace po = boost::program_options;
using Arm = ompl::robots::Reachy2;
using Robot = ompl::robots::HolonomicMobileManipulator<Arm>;
using Configuration = Robot::Configuration;
using Operations = Robot::ConfigurationOperations;
constexpr int N = static_cast<int>(Robot::nJoints);

namespace
{
    struct Box
    {
        Eigen::Vector3d center;
        Eigen::Vector3d half;
    };

    double boxDistance(const Eigen::Vector3d &point, const Box &box)
    {
        const Eigen::Vector3d q = (point - box.center).cwiseAbs() - box.half;
        return q.cwiseMax(0.0).norm() + std::min(q.maxCoeff(), 0.0);
    }

    std::vector<Box> shelf()
    {
        constexpr double bottom = 0.9144, x = 0.62, depth = 0.14, width = 0.75;
        constexpr double pitch = 0.44, panelHalf = 0.46;
        constexpr double top = bottom + 2.0 * panelHalf;
        std::vector<Box> boxes{{{x + depth, 0, top / 2}, {0.02, width / 2, top / 2}},
                               {{x, width / 2, top / 2}, {depth, 0.02, top / 2}},
                               {{x, -width / 2, top / 2}, {depth, 0.02, top / 2}}};
        for (double z : {0.0, pitch, 2.0 * pitch})
            boxes.push_back({{x, 0, bottom + z}, {depth, width / 2, 0.015}});
        return boxes;
    }

    class MobileBarrier
    {
    public:
        static constexpr double worldMargin = 0.010;
        static constexpr double selfMargin = 0.005;
        static constexpr int maxConstraints = Robot::nSpheres + Robot::nSelfPairs;
        using Values = Eigen::Matrix<double, maxConstraints, 1>;
        using Rows = Eigen::Matrix<double, maxConstraints, N>;
        using Centers = Robot::SphereCenters;

        struct Evaluation
        {
            Values values;
            Rows rows;
            Eigen::Matrix<int, maxConstraints, 1> constraint;
            Eigen::Matrix<double, Robot::nSpheres, 1> boundary;
            int active{0};
            bool inBounds{true};
        };

        MobileBarrier(const Robot &robot, const ompl::sdf::GridSDF &field,
                      const Configuration &reference)
          : robot_(robot), field_(field)
        {
            buildLeverBounds();
            const auto kin = robot_.kinematics(reference);
            for (std::size_t p = 0; p < Robot::nSelfPairs; ++p)
            {
                const auto pair = Robot::selfPairs()[p];
                const auto &a = Robot::spheres()[pair.a];
                const auto &b = Robot::spheres()[pair.b];
                const double gap = (Robot::sphereCenter(kin, pair.a) -
                                    Robot::sphereCenter(kin, pair.b)).norm() -
                                   a.radius - b.radius;
                // Invariant pairs need no derivative row; unsafe invariant geometry is
                // still rejected by safe().
                if (a.influence != b.influence && gap > selfMargin + 0.02)
                    selfPairs_.push_back(p);
            }
            buildPairLeverBounds();
        }

        std::size_t enabledSelfPairs() const
        {
            return selfPairs_.size();
        }

        void decreaseRates(const Configuration &speed, Values &rates) const
        {
            rates.setZero();
            const Configuration absolute = speed.cwiseAbs();
            rates.template head<Robot::nSpheres>() =
                field_.maxGradientNorm() * (leverBounds_ * absolute);
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
                rates[Robot::nSpheres + p] = pairLeverBounds_.row(p).dot(absolute);
        }

        void evaluateScreened(const Configuration &q, const Values &threshold,
                              Evaluation &out) const
        {
            const auto kin = robot_.kinematics(q);
            Centers centers;
            out.active = 0;
            out.inBounds = true;
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers.col(i) = Robot::sphereCenter(kin, i);
                out.inBounds = out.inBounds && field_.inBounds(centers.col(i));
                out.boundary[i] = boundaryClearance(centers.col(i));
                out.values[i] = field_.distance(centers.col(i)) -
                                Robot::spheres()[i].radius - worldMargin;
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const auto pair = Robot::selfPairs()[selfPairs_[p]];
                out.values[Robot::nSpheres + p] =
                    (centers.col(pair.a) - centers.col(pair.b)).norm() -
                    Robot::spheres()[pair.a].radius - Robot::spheres()[pair.b].radius -
                    selfMargin;
            }
            const std::size_t count = Robot::nSpheres + selfPairs_.size();
            for (std::size_t flat = 0; flat < count; ++flat)
            {
                if (out.values[flat] > threshold[flat])
                    continue;
                const int row = out.active++;
                out.constraint[row] = static_cast<int>(flat);
                if (flat < Robot::nSpheres)
                    out.rows.row(row) = field_.gradient(centers.col(flat)).transpose() *
                                        Robot::sphereJacobian(kin, flat);
                else
                    out.rows.row(row) = pairGradient(
                        kin, centers, flat - Robot::nSpheres).transpose();
            }
        }

        double certifiedDuration(const Evaluation &evaluation, const Configuration &control,
                                 double gamma, double buffer = 0.0) const
        {
            const Configuration speed = control.cwiseAbs();
            const auto worldTravel = (leverBounds_ * speed).eval();
            const double lipschitz = std::max(field_.maxGradientNorm(), 1.0);
            double duration = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                if (worldTravel[i] <= 0.0)
                    continue;
                const double allowance = std::min(
                    gamma * std::max(evaluation.values[i] - buffer, 0.0) / lipschitz,
                    evaluation.boundary[i]);
                duration = std::min(duration, allowance / worldTravel[i]);
            }
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const double travel = pairLeverBounds_.row(p).dot(speed);
                if (travel > 0.0)
                    duration = std::min(duration,
                        gamma * std::max(evaluation.values[Robot::nSpheres + p] - buffer, 0.0) /
                            travel);
            }
            return std::max(duration, 0.0);
        }

        bool safe(const Configuration &q, double *world = nullptr, double *self = nullptr) const
        {
            const auto kin = robot_.kinematics(q);
            Centers centers;
            bool inBounds = true;
            double worstWorld = std::numeric_limits<double>::infinity();
            double worstSelf = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                centers.col(i) = Robot::sphereCenter(kin, i);
                inBounds = inBounds && field_.inBounds(centers.col(i));
                worstWorld = std::min(worstWorld, field_.distance(centers.col(i)) -
                    Robot::spheres()[i].radius - worldMargin);
            }
            for (const std::size_t source : selfPairs_)
            {
                const auto pair = Robot::selfPairs()[source];
                worstSelf = std::min(worstSelf,
                    (centers.col(pair.a) - centers.col(pair.b)).norm() -
                    Robot::spheres()[pair.a].radius - Robot::spheres()[pair.b].radius -
                    selfMargin);
            }
            if (world)
                *world = worstWorld;
            if (self)
                *self = worstSelf;
            return inBounds && worstWorld >= 0.0 && worstSelf >= 0.0;
        }

    private:
        double boundaryClearance(const Eigen::Vector3d &point) const
        {
            return std::min((point - field_.bounds().min()).minCoeff(),
                            (field_.bounds().max() - point).minCoeff());
        }

        void buildLeverBounds()
        {
            leverBounds_.setZero();
            const auto &steps = Arm::steps();
            for (std::size_t i = 0; i < Robot::nSpheres; ++i)
            {
                leverBounds_(i, 0) = leverBounds_(i, 1) = 1.0;
                int link = Robot::spheres()[i].link;
                double reach = Eigen::Vector3d(Robot::spheres()[i].center.data()).norm();
                while (link > 0)
                {
                    const auto &step = steps[link - 1];
                    if (step.active >= 0)
                        leverBounds_(i, 3 + step.active) = reach;
                    reach += Eigen::Vector3d(step.xyz.data()).norm();
                    link = step.parent;
                }
                // Rotation about the base z axis moves a point no faster than its
                // full chain distance from base_link.
                leverBounds_(i, 2) = reach;
            }
        }

        void buildPairLeverBounds()
        {
            pairLeverBounds_.setZero();
            for (std::size_t p = 0; p < selfPairs_.size(); ++p)
            {
                const auto pair = Robot::selfPairs()[selfPairs_[p]];
                const auto a = Robot::spheres()[pair.a].influence;
                const auto b = Robot::spheres()[pair.b].influence;
                for (int j = 0; j < static_cast<int>(Arm::nJoints); ++j)
                {
                    const bool movesA = (a & (1u << j)) != 0;
                    const bool movesB = (b & (1u << j)) != 0;
                    if (movesA != movesB)
                        pairLeverBounds_(p, 3 + j) =
                            leverBounds_(movesA ? pair.a : pair.b, 3 + j);
                }
                // All planar-base contributions are rigid and cancel exactly.
                pairLeverBounds_.template block<1, 3>(p, 0).setZero();
            }
        }

        Configuration pairGradient(const Robot::Kinematics &kin, const Centers &centers,
                                   std::size_t enabled) const
        {
            const auto pair = Robot::selfPairs()[selfPairs_[enabled]];
            const Eigen::Vector3d delta = centers.col(pair.a) - centers.col(pair.b);
            Configuration row = Configuration::Zero();
            if (delta.norm() > 1e-12)
                row = (Robot::sphereJacobian(kin, pair.a) -
                       Robot::sphereJacobian(kin, pair.b)).transpose() * delta.normalized();
            row.template head<3>().setZero();
            return row;
        }

        const Robot &robot_;
        const ompl::sdf::GridSDF &field_;
        std::vector<std::size_t> selfPairs_;
        Eigen::Matrix<double, Robot::nSpheres, N> leverBounds_;
        Eigen::Matrix<double, Robot::nSelfPairs, N> pairLeverBounds_;
    };

    class MobileFilter : public ompl::cbf::RobotControlFilter<Robot>
    {
    public:
        using Base = ompl::cbf::RobotControlFilter<Robot>;
        using Status = Base::Status;
        static constexpr double gamma = 0.6;
        static constexpr double integrationBuffer = 0.001;

        MobileFilter(const MobileBarrier &barrier, const Configuration &lower,
                     const Configuration &upper)
          : barrier_(barrier), lowerPosition_(lower), upperPosition_(upper)
        {
            rowUpper_.setConstant(std::numeric_limits<double>::infinity());
            barrier_.decreaseRates(maxSpeed(), decreaseRates_);
            const Configuration speed = maxSpeed();
            inverseSquared_ = speed.cwiseInverse().cwiseProduct(speed.cwiseInverse());
        }

        static Configuration maxSpeed()
        {
            return Robot::velocityLimits();
        }

        Status filter(const Configuration &q, const Configuration &nominal, double dt,
                      Configuration &filtered) const override
        {
            double certified = 0.0;
            return filter(q, nominal, dt, filtered, certified);
        }

        Status filter(const Configuration &q, const Configuration &nominal, double dt,
                      Configuration &filtered, double &certified) const override
        {
            ++calls_;
            certified = 0.0;
            if (dt <= 0.0)
            {
                filtered.setZero();
                return Status::Blocked;
            }
            threshold_ = decreaseRates_ * dt;
            threshold_.array() += integrationBuffer;
            barrier_.evaluateScreened(q, threshold_, evaluation_);
            activeRows_ += evaluation_.active;
            if (!evaluation_.inBounds)
            {
                filtered.setZero();
                return Status::Blocked;
            }

            Configuration lower = -maxSpeed(), upper = maxSpeed();
            for (int j = 0; j < N; ++j)
            {
                if (j == 2)
                    continue;
                lower[j] = std::max(lower[j], (lowerPosition_[j] - q[j]) / dt);
                upper[j] = std::min(upper[j], (upperPosition_[j] - q[j]) / dt);
                if (lower[j] > upper[j])
                    lower[j] = upper[j] = 0.0;
            }

            const Eigen::Index active = evaluation_.active;
            if (active == 0)
                filtered = nominal.cwiseMax(lower).cwiseMin(upper);
            else
            {
                ++qpCalls_;
                hessian_.setZero();
                hessian_.diagonal() = inverseSquared_;
                objective_ = -inverseSquared_.cwiseProduct(nominal);
                for (Eigen::Index row = 0; row < active; ++row)
                    rowLower_[row] = -gamma *
                        (evaluation_.values[evaluation_.constraint[row]] - integrationBuffer) / dt;
                try
                {
                    if (solver_.solve(filtered, hessian_, objective_, lower, upper,
                                      evaluation_.rows.topRows(active), rowLower_.head(active),
                                      rowUpper_.head(active)) != Solver::OK)
                    {
                        filtered.setZero();
                        return Status::Blocked;
                    }
                }
                catch (const std::exception &)
                {
                    filtered.setZero();
                    return Status::Blocked;
                }
            }

            certified = barrier_.certifiedDuration(
                evaluation_, filtered, gamma, integrationBuffer);
            for (int j = 0; j < N; ++j)
            {
                if (j == 2 || filtered[j] == 0.0)
                    continue;
                const double room = (filtered[j] > 0.0 ? upperPosition_[j] : lowerPosition_[j]) - q[j];
                certified = std::min(certified, std::max(room / filtered[j], 0.0));
            }
            return (filtered - nominal).norm() <= 1e-12 ? Status::Unchanged : Status::Filtered;
        }

        const char *name() const override { return "reachy2-mobile-cbf-qp"; }
        std::size_t calls() const { return calls_; }
        std::size_t qpCalls() const { return qpCalls_; }
        double meanActiveRows() const
        {
            return calls_ ? static_cast<double>(activeRows_) / calls_ : 0.0;
        }

    private:
        using Solver = qpmad::SolverTemplate<double, N, 1, MobileBarrier::maxConstraints>;
        const MobileBarrier &barrier_;
        Configuration lowerPosition_, upperPosition_;
        mutable Solver solver_;
        mutable Eigen::Matrix<double, N, N> hessian_;
        Configuration inverseSquared_;
        mutable Configuration objective_;
        mutable MobileBarrier::Values rowLower_, rowUpper_, threshold_;
        MobileBarrier::Values decreaseRates_;
        mutable MobileBarrier::Evaluation evaluation_;
        mutable std::size_t calls_{0}, qpCalls_{0}, activeRows_{0};
    };

    bool solveArmIK(const Robot &robot, const MobileBarrier &barrier,
                    const Eigen::Vector3d &left, const Eigen::Vector3d &right,
                    Configuration &q, std::mt19937 &rng,
                    const Configuration *guess = nullptr)
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const auto lo = Arm::lowerBounds(), hi = Arm::upperBounds();
        const Configuration fixedBase = q;
        double best = std::numeric_limits<double>::infinity();
        Configuration bestQ = q;
        for (int restart = 0; restart < 80; ++restart)
        {
            q.template head<3>() = fixedBase.template head<3>();
            if (restart == 0 && guess)
                q.template tail<Arm::nJoints>() = guess->template tail<Arm::nJoints>();
            else
                for (int j = 0; j < static_cast<int>(Arm::nJoints); ++j)
                    q[3 + j] = lo[j] + unit(rng) * (hi[j] - lo[j]);

            for (int iteration = 0; iteration < 150; ++iteration)
            {
                const auto kin = robot.kinematics(q);
                Eigen::Matrix<double, 6, 1> error;
                error.template head<3>() = left - Robot::tipPosition(kin, true);
                error.template tail<3>() = right - Robot::tipPosition(kin, false);
                if (error.norm() < 0.012 && barrier.safe(q))
                    return true;
                Eigen::Matrix<double, 6, Arm::nJoints> jacobian;
                jacobian.template topRows<3>() =
                    Robot::tipJacobian(kin, true).template rightCols<Arm::nJoints>();
                jacobian.template bottomRows<3>() =
                    Robot::tipJacobian(kin, false).template rightCols<Arm::nJoints>();
                constexpr double damping = 2e-3;
                const auto dq = jacobian.transpose() *
                    (jacobian * jacobian.transpose() +
                     damping * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(error);
                q.template tail<Arm::nJoints>() += 0.45 * dq;
                q.template tail<Arm::nJoints>() =
                    q.template tail<Arm::nJoints>().cwiseMax(lo).cwiseMin(hi);
                if (error.norm() < best && barrier.safe(q))
                {
                    best = error.norm();
                    bestQ = q;
                }
            }
        }
        q = bestQ;
        return best < 0.05;
    }

    class BiasedSampler : public ob::StateSampler
    {
    public:
        BiasedSampler(const ob::StateSpace *space, std::uint_fast32_t seed,
                      const Configuration &transit,
                      const std::vector<Configuration> &goals, bool uniformOnly = false)
          : ob::StateSampler(space), transit_(transit), goals_(goals), uniformOnly_(uniformOnly)
        {
            rng_.setLocalSeed(seed);
        }

        void sampleUniform(ob::State *state) override
        {
            Configuration q;
            const auto &bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
            const double branch = uniformOnly_ ? 1.0 : rng_.uniform01();
            if (branch < 0.60)
            {
                q = transit_;
                for (int j = 0; j < 3; ++j)
                    q[j] = rng_.uniformReal(bounds.low[j], bounds.high[j]);
            }
            else if (branch < 0.80)
            {
                q = goals_[rng_.uniformInt(0, goals_.size() - 1)];
                q[0] += rng_.gaussian(0.0, 0.08);
                q[1] += rng_.gaussian(0.0, 0.08);
                q[2] += rng_.gaussian(0.0, 0.15);
                for (int j = 3; j < N; ++j)
                    q[j] += rng_.gaussian(0.0, 0.12);
            }
            else
                for (int j = 0; j < N; ++j)
                    q[j] = rng_.uniformReal(bounds.low[j], bounds.high[j]);
            auto *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
            for (int j = 0; j < N; ++j)
                values[j] = q[j];
            space_->enforceBounds(state);
        }

        void sampleUniformNear(ob::State *state, const ob::State *, double) override
        {
            sampleUniform(state);
        }
        void sampleGaussian(ob::State *state, const ob::State *, double) override
        {
            sampleUniform(state);
        }

    private:
        Configuration transit_;
        const std::vector<Configuration> &goals_;
        bool uniformOnly_;
    };

    class SeededRRTConnect : public og::RRTConnect
    {
    public:
        SeededRRTConnect(const ob::SpaceInformationPtr &si, std::uint_fast32_t seed)
          : og::RRTConnect(si)
        {
            rng_.setLocalSeed(seed);
        }
    };

    Configuration stateConfiguration(const ob::State *state)
    {
        Configuration q;
        const double *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
        for (int j = 0; j < N; ++j)
            q[j] = values[j];
        return q;
    }

    ob::GoalPtr makeGoals(const ob::SpaceInformationPtr &si,
                          const std::vector<Configuration> &goals)
    {
        auto result = std::make_shared<ob::GoalStates>(si);
        ob::ScopedState<> state(si->getStateSpace());
        for (const Configuration &q : goals)
        {
            for (int j = 0; j < N; ++j)
                state[j] = q[j];
            result->addState(state.get());
        }
        result->setThreshold(0.08);
        return result;
    }

    struct Result
    {
        bool solved{false};
        double seconds{0.0};
        std::size_t evaluations{0}, nodes{0}, qpCalls{0}, coarse{0}, misses{0};
        double meanRows{0.0}, worstWorld{0.0}, worstSelf{0.0};
        double leftError{0.0}, rightError{0.0};
        std::vector<Configuration> motion;
    };

    void denseAudit(Result &result, const Robot &robot, const MobileBarrier &barrier,
                    const Eigen::Vector3d &left, const Eigen::Vector3d &right,
                    double spacing)
    {
        if (result.motion.empty())
            return;
        std::vector<Configuration> dense{result.motion.front()};
        for (std::size_t i = 1; i < result.motion.size(); ++i)
        {
            const double span = Operations::distance(result.motion[i - 1], result.motion[i]);
            const int parts = std::max(1, static_cast<int>(std::ceil(span / spacing)));
            for (int part = 1; part <= parts; ++part)
                dense.push_back(Operations::interpolate(result.motion[i - 1], result.motion[i],
                    static_cast<double>(part) / parts));
        }
        result.motion = std::move(dense);
        result.worstWorld = result.worstSelf = std::numeric_limits<double>::infinity();
        result.solved = true;
        for (const Configuration &q : result.motion)
        {
            double world = 0.0, self = 0.0;
            barrier.safe(q, &world, &self);
            result.worstWorld = std::min(result.worstWorld, world);
            result.worstSelf = std::min(result.worstSelf, self);
            result.solved = result.solved && world >= -1e-9 && self >= -1e-9;
        }
        const auto kin = robot.kinematics(result.motion.back());
        result.leftError = (Robot::tipPosition(kin, true) - left).norm();
        result.rightError = (Robot::tipPosition(kin, false) - right).norm();
        result.solved = result.solved && result.leftError <= 0.04 && result.rightError <= 0.04;
    }

    template <typename Space>
    void fillState(const std::shared_ptr<Space> &space, const Configuration &q,
                   ob::ScopedState<> &state)
    {
        (void)space;
        for (int j = 0; j < N; ++j)
            state[j] = q[j];
    }

    Result runBaseline(const Robot &robot, const MobileBarrier &barrier,
                       const Configuration &start, const std::vector<Configuration> &goals,
                       const Configuration &transit, const Configuration &lower,
                       const Configuration &upper, const Eigen::Vector3d &left,
                       const Eigen::Vector3d &right, double limit, std::uint_fast32_t seed,
                       double auditSpacing, bool uniformSampler)
    {
        auto space = std::make_shared<ompl::cbf::RobotStateSpace<Robot>>(
            Robot::velocityLimits(), lower, upper);
        space->setLongestValidSegmentFraction(auditSpacing / space->getMaximumExtent());
        space->setStateSamplerAllocator([seed, transit, &goals, uniformSampler](const ob::StateSpace *s)
        { return std::make_shared<BiasedSampler>(s, seed + 1, transit, goals, uniformSampler); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        std::size_t checks = 0;
        si->setStateValidityChecker([&](const ob::State *state)
        { ++checks; return barrier.safe(stateConfiguration(state)); });
        si->setup();
        auto definition = std::make_shared<ob::ProblemDefinition>(si);
        ob::ScopedState<> startState(space);
        fillState(space, start, startState);
        definition->addStartState(startState.get());
        definition->setGoal(makeGoals(si, goals));
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(0.8);
        planner->setProblemDefinition(definition);
        planner->setup();
        const auto begin = std::chrono::steady_clock::now();
        const auto status = planner->solve(ob::timedPlannerTerminationCondition(limit));
        Result result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.evaluations = checks;
        ob::PlannerData data(si); planner->getPlannerData(data); result.nodes = data.numVertices();
        if (!status || !definition->hasSolution())
            return result;
        auto path = std::static_pointer_cast<og::PathGeometric>(definition->getSolutionPath());
        const int count = std::max(2, static_cast<int>(std::ceil(path->length() / auditSpacing)) + 1);
        path->interpolate(count);
        for (std::size_t i = 0; i < path->getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(path->getState(i)));
        denseAudit(result, robot, barrier, left, right, auditSpacing);
        return result;
    }

    Result runCBF(const Robot &robot, const MobileBarrier &barrier,
                  const Configuration &start, const std::vector<Configuration> &goals,
                  const Configuration &transit, const Configuration &lower,
                  const Configuration &upper, const Eigen::Vector3d &left,
                  const Eigen::Vector3d &right, double limit, std::uint_fast32_t seed,
                  double auditSpacing, bool uniformSampler)
    {
        MobileFilter filter(barrier, lower, upper);
        using Space = ompl::cbf::RobotFilteredStateSpace<Robot>;
        auto space = std::make_shared<Space>(filter, 0.01, Robot::velocityLimits(), lower, upper);
        space->setStateSamplerAllocator([seed, transit, &goals, uniformSampler](const ob::StateSpace *s)
        { return std::make_shared<BiasedSampler>(s, seed + 1, transit, goals, uniformSampler); });
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::RobotFilteredMotionValidator<Robot>>(si));
        si->setup();
        auto definition = std::make_shared<ob::ProblemDefinition>(si);
        ob::ScopedState<> startState(space);
        fillState(space, start, startState);
        definition->addStartState(startState.get());
        definition->setGoal(makeGoals(si, goals));
        auto planner = std::make_shared<SeededRRTConnect>(si, seed + 2);
        planner->setRange(0.8);
        planner->setProblemDefinition(definition);
        planner->setup();
        const auto begin = std::chrono::steady_clock::now();
        const auto status = planner->solve(ob::timedPlannerTerminationCondition(limit));
        Result result;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const auto stats = space->statistics();
        result.evaluations = stats.steps; result.coarse = stats.coarse;
        result.qpCalls = filter.qpCalls(); result.meanRows = filter.meanActiveRows();
        ob::PlannerData data(si); planner->getPlannerData(data); result.nodes = data.numVertices();
        if (!status || !definition->hasSolution())
            return result;
        const auto path = std::static_pointer_cast<og::PathGeometric>(definition->getSolutionPath());
        const auto executed = ompl::cbf::robotExecutedPath<Robot>(*path, auditSpacing, &result.misses);
        for (std::size_t i = 0; i < executed.getStateCount(); ++i)
            result.motion.push_back(stateConfiguration(executed.getState(i)));
        denseAudit(result, robot, barrier, left, right, auditSpacing);
        return result;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
    }

    struct Travel
    {
        double base{0.0}, yaw{0.0}, arm{0.0}, duration{0.0};
    };

    Travel travel(const std::vector<Configuration> &motion)
    {
        Travel out;
        for (std::size_t i = 1; i < motion.size(); ++i)
        {
            const Configuration delta = Operations::difference(motion[i - 1], motion[i]);
            out.base += delta.template head<2>().norm();
            out.yaw += std::abs(delta[2]);
            out.arm += delta.template tail<Arm::nJoints>().norm();
            out.duration += Operations::duration(motion[i - 1], motion[i], Robot::velocityLimits());
        }
        return out;
    }

    void writePath(const std::string &path, const std::vector<Configuration> &motion)
    {
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("cannot open output path " + path);
        out.precision(17);
        out << "# trajectory time_s base_x base_y base_yaw";
        for (const char *name : Arm::jointNames())
            out << ' ' << name;
        out << '\n';
        double time = 0.0;
        for (std::size_t i = 0; i < motion.size(); ++i)
        {
            if (i)
                time += Operations::duration(motion[i - 1], motion[i], Robot::velocityLimits());
            out << time;
            for (int j = 0; j < N; ++j)
                out << ' ' << motion[i][j];
            out << '\n';
        }
    }

    std::string baselinePathFor(const std::string &cbf)
    {
        const std::string suffix = "_cbf.path";
        if (cbf.size() >= suffix.size() && cbf.compare(cbf.size() - suffix.size(), suffix.size(), suffix) == 0)
            return cbf.substr(0, cbf.size() - suffix.size()) + "_ompl.path";
        return cbf + ".ompl";
    }
}  // namespace

int main(int argc, char **argv)
{
    double seconds, shortcut, startX, startY, startYaw, xmin, xmax, ymin, ymax;
    int trials;
    bool smoke{false};
    bool uniformSampler{false};
    std::string output;
    std::vector<double> leftValues, rightValues;
    po::options_description options("Reachy2 mobile RRT/CBF options");
    options.add_options()
        ("help,h", "show this help")
        ("smoke", po::bool_switch(&smoke), "enforce deterministic three-trial acceptance checks")
        ("uniform-sampler", po::bool_switch(&uniformSampler),
         "disable the transit/goal bias and sample fully uniformly")
        ("seconds", po::value<double>(&seconds)->default_value(10.0), "solve time per row and trial")
        ("trials", po::value<int>(&trials)->default_value(3), "paired trial count")
        ("shortcut", po::value<double>(&shortcut)->default_value(0.10), "rope anchor spacing in seconds (0 disables)")
        ("output,o", po::value<std::string>(&output)->default_value("reachy2_mobile_cbf.path"), "CBF trajectory path")
        ("start-x", po::value<double>(&startX)->default_value(-0.70), "base start x")
        ("start-y", po::value<double>(&startY)->default_value(0.0), "base start y")
        ("start-yaw", po::value<double>(&startYaw)->default_value(0.0), "base start yaw")
        ("xmin", po::value<double>(&xmin)->default_value(-1.0), "base x lower bound")
        ("xmax", po::value<double>(&xmax)->default_value(0.35), "base x upper bound")
        ("ymin", po::value<double>(&ymin)->default_value(-0.65), "base y lower bound")
        ("ymax", po::value<double>(&ymax)->default_value(0.65), "base y upper bound")
        ("left-goal", po::value<std::vector<double>>(&leftValues)->multitoken()->default_value(
            {0.62, 0.20, 1.1344}, "0.62 0.20 1.1344"), "left Cartesian goal x y z")
        ("right-goal", po::value<std::vector<double>>(&rightValues)->multitoken()->default_value(
            {0.62, -0.20, 1.1344}, "0.62 -0.20 1.1344"), "right Cartesian goal x y z");
    po::variables_map variables;
    try
    {
        po::store(po::parse_command_line(argc, argv, options), variables);
        po::notify(variables);
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
    if (variables.count("help"))
    {
        std::cout << options << '\n';
        return 0;
    }
    if (seconds <= 0.0 || trials < 1 || shortcut < 0.0 || xmin >= xmax || ymin >= ymax ||
        leftValues.size() != 3 || rightValues.size() != 3)
    {
        std::fprintf(stderr, "invalid limits, bounds, shortcut, trials, or Cartesian goal\n");
        return 2;
    }

    ompl::RNG::setSeed(7);
    const Eigen::Vector3d left(leftValues.data()), right(rightValues.data());
    const auto boxes = shelf();
    const auto distance = [&boxes](const Eigen::Vector3d &point)
    {
        double result = std::numeric_limits<double>::infinity();
        for (const Box &box : boxes)
            result = std::min(result, boxDistance(point, box));
        return result;
    };
    std::printf("Baking mobile shelf SDF...\n");
    const ompl::sdf::GridSDF field(distance,
        Eigen::AlignedBox3d(Eigen::Vector3d(-2.0, -1.7, -0.5),
                            Eigen::Vector3d(1.7, 1.7, 2.4)), 0.025);
    const Robot robot;
    Configuration lower = Robot::lowerBounds(), upper = Robot::upperBounds();
    lower[0] = xmin; upper[0] = xmax; lower[1] = ymin; upper[1] = ymax;
    Configuration reference = Configuration::Zero();
    reference.template tail<Arm::nJoints>() = 0.5 * (Arm::lowerBounds() + Arm::upperBounds());
    const MobileBarrier barrier(robot, field, reference);

    std::mt19937 transitRng(7);
    Configuration transit = reference;
    transit.template head<3>().setZero();
    if (!solveArmIK(robot, barrier, Eigen::Vector3d(0.25, 0.20, 1.1344),
                    Eigen::Vector3d(0.25, -0.20, 1.1344), transit, transitRng))
    {
        std::fprintf(stderr, "could not construct collision-free transit posture\n");
        return 3;
    }
    Configuration start = transit;
    start.template head<3>() << startX, startY, Operations::wrapYaw(startYaw);
    double startWorld = 0.0, startSelf = 0.0;
    if (!barrier.safe(start, &startWorld, &startSelf))
    {
        std::fprintf(stderr, "start is unsafe: world=%+.4f self=%+.4f\n", startWorld, startSelf);
        return 3;
    }

    // Deterministic mobile IK goal pool, deliberately outside every timed region.
    std::vector<Configuration> goals;
    std::mt19937 ikRng(71821);
    std::uniform_real_distribution<double> gx(-0.18, 0.18), gy(-0.30, 0.30),
        gyaw(-0.55, 0.55);
    for (int attempt = 0; attempt < 240 && goals.size() < 16; ++attempt)
    {
        Configuration q = transit;
        q.template head<3>() << gx(ikRng), gy(ikRng), gyaw(ikRng);
        if (q[0] < xmin || q[0] > xmax || q[1] < ymin || q[1] > ymax)
            continue;
        if (solveArmIK(robot, barrier, left, right, q, ikRng, &transit))
            goals.push_back(q);
    }
    if (goals.size() < 4)
    {
        std::fprintf(stderr, "mobile IK produced only %zu valid goals (need at least 4); bounds x=[%.2f,%.2f] y=[%.2f,%.2f]\n",
                     goals.size(), xmin, xmax, ymin, ymax);
        return 4;
    }
    std::printf("Reachy2 mobile: %d DoF, %zu spheres, %zu/%zu self pairs, %zu IK goals\n",
                N, Robot::nSpheres, barrier.enabledSelfPairs(), Robot::nSelfPairs, goals.size());

    constexpr double auditSpacing = 0.01;
    std::vector<double> omplTimes, cbfTimes, omplEvals, cbfEvals, omplNodes, cbfNodes;
    std::size_t omplSolved = 0, cbfSolved = 0;
    Result omplRepresentative, cbfRepresentative;
    for (int trial = 0; trial < trials; ++trial)
    {
        const std::uint_fast32_t seed = 9001u + 31u * trial;
        Result ompl, cbf;
        if (trial % 2)
        {
            ompl = runBaseline(robot, barrier, start, goals, transit, lower, upper,
                               left, right, seconds, seed, auditSpacing, uniformSampler);
            cbf = runCBF(robot, barrier, start, goals, transit, lower, upper,
                         left, right, seconds, seed, auditSpacing, uniformSampler);
        }
        else
        {
            cbf = runCBF(robot, barrier, start, goals, transit, lower, upper,
                         left, right, seconds, seed, auditSpacing, uniformSampler);
            ompl = runBaseline(robot, barrier, start, goals, transit, lower, upper,
                               left, right, seconds, seed, auditSpacing, uniformSampler);
        }
        std::printf("trial %d: OMPL %.2f ms %zu eval %zu nodes %s | CBF %.2f ms %zu eval %zu nodes %s\n",
            trial + 1, 1e3 * ompl.seconds, ompl.evaluations, ompl.nodes, ompl.solved ? "ok" : "fail",
            1e3 * cbf.seconds, cbf.evaluations, cbf.nodes, cbf.solved ? "ok" : "fail");
        omplTimes.push_back(ompl.seconds); cbfTimes.push_back(cbf.seconds);
        omplEvals.push_back(ompl.evaluations); cbfEvals.push_back(cbf.evaluations);
        omplNodes.push_back(ompl.nodes); cbfNodes.push_back(cbf.nodes);
        if (ompl.solved) { ++omplSolved; omplRepresentative = std::move(ompl); }
        if (cbf.solved) { ++cbfSolved; cbfRepresentative = std::move(cbf); }
    }
    std::printf("median: OMPL %.2f ms %.0f eval %.0f nodes, solved %zu/%d | CBF %.2f ms %.0f eval %.0f nodes, solved %zu/%d\n",
        1e3 * median(omplTimes), median(omplEvals), median(omplNodes), omplSolved, trials,
        1e3 * median(cbfTimes), median(cbfEvals), median(cbfNodes), cbfSolved, trials);
    if (!omplSolved || !cbfSolved)
    {
        std::fprintf(stderr, "both planner rows must produce at least one audited path\n");
        return 5;
    }

    double omplBefore = travel(omplRepresentative.motion).duration;
    double cbfBefore = travel(cbfRepresentative.motion).duration;
    ompl::cbf::ShortcutReport cbfShortcut;
    if (shortcut > 0.0)
    {
        auto space = std::make_shared<ompl::cbf::RobotStateSpace<Robot>>(
            Robot::velocityLimits(), lower, upper);
        // Rope may create a clearance minimum between the phase-shifted samples used by
        // its discrete motion check. Check twice as densely and reserve 1 mm of the
        // modeled margin so the independent post-shortcut audit cannot expose a tiny
        // between-sample crossing.
        space->setLongestValidSegmentFraction(
            (0.5 * auditSpacing) / space->getMaximumExtent());
        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker([&](const ob::State *state)
        {
            double world = 0.0, self = 0.0;
            return barrier.safe(stateConfiguration(state), &world, &self) &&
                   world >= 0.001 && self >= 0.001;
        });
        si->setup();
        og::PathGeometric path(si);
        ob::ScopedState<> state(space);
        for (const Configuration &q : omplRepresentative.motion)
        {
            fillState(space, q, state);
            path.append(state.get());
        }
        og::PathSimplifier(si).ropeShortcutPath(path, shortcut, 0.01);
        omplRepresentative.motion.clear();
        for (std::size_t i = 0; i < path.getStateCount(); ++i)
            omplRepresentative.motion.push_back(stateConfiguration(path.getState(i)));

        MobileFilter shortcutFilter(barrier, lower, upper);
        ompl::cbf::RobotFilteredStateSpace<Robot> shortcutSpace(
            shortcutFilter, 0.01, Robot::velocityLimits(), lower, upper);
        const auto rollout = [&shortcutSpace](const Configuration &from, const Configuration &to)
        {
            auto result = shortcutSpace.roll(from, to, 1.0);
            return result.reachedTarget ? std::move(result.waypoints) : std::vector<Configuration>();
        };
        cbfRepresentative.motion = ompl::cbf::ropeShortcut(
            cbfRepresentative.motion, shortcut, 1e-6, rollout, &cbfShortcut, 0.01,
            Operations());
    }
    denseAudit(omplRepresentative, robot, barrier, left, right, auditSpacing);
    denseAudit(cbfRepresentative, robot, barrier, left, right, auditSpacing);
    if (!omplRepresentative.solved || !cbfRepresentative.solved)
    {
        std::fprintf(stderr,
            "shortened path failed dense audit: OMPL=%s world=%+.7f self=%+.7f L=%.5f R=%.5f; "
            "CBF=%s world=%+.7f self=%+.7f L=%.5f R=%.5f\n",
            omplRepresentative.solved ? "ok" : "fail", omplRepresentative.worstWorld,
            omplRepresentative.worstSelf, omplRepresentative.leftError,
            omplRepresentative.rightError, cbfRepresentative.solved ? "ok" : "fail",
            cbfRepresentative.worstWorld, cbfRepresentative.worstSelf,
            cbfRepresentative.leftError, cbfRepresentative.rightError);
        return 6;
    }

    if (smoke)
    {
        const double omplDisplacement =
            (omplRepresentative.motion.back().template head<2>() - start.template head<2>()).norm();
        const double cbfDisplacement =
            (cbfRepresentative.motion.back().template head<2>() - start.template head<2>()).norm();
        if (trials != 3 || omplSolved != 3 || cbfSolved != 3 ||
            omplDisplacement < 0.30 || cbfDisplacement < 0.30 ||
            cbfRepresentative.misses != 0)
        {
            std::fprintf(stderr,
                "smoke acceptance failed: trials=%d solved=%zu/%zu displacement=%.3f/%.3f misses=%zu\n",
                trials, omplSolved, cbfSolved, omplDisplacement, cbfDisplacement,
                cbfRepresentative.misses);
            return 8;
        }
    }

    const Travel omplTravel = travel(omplRepresentative.motion);
    const Travel cbfTravel = travel(cbfRepresentative.motion);
    const auto report = [](const char *name, const Result &result, const Travel &motion,
                           double before)
    {
        std::printf("%-10s base %.3f m yaw %.3f rad arm %.3f rad duration %.3f s shortcut %.3f->%.3f s clearances world=%+.4f self=%+.4f errors L=%.4f R=%.4f\n",
            name, motion.base, motion.yaw, motion.arm, motion.duration, before, motion.duration,
            result.worstWorld, result.worstSelf, result.leftError, result.rightError);
    };
    report("ompl-rrtc", omplRepresentative, omplTravel, omplBefore);
    report("cbf-rrtc", cbfRepresentative, cbfTravel, cbfBefore);
    std::printf("CBF QP %zu/%zu, %.2f rows/call, %zu coarse steps, %zu replay misses; shortcut %zu/%zu accepted, max gap %.3g\n",
        cbfRepresentative.qpCalls, cbfRepresentative.evaluations, cbfRepresentative.meanRows,
        cbfRepresentative.coarse, cbfRepresentative.misses, cbfShortcut.accepted,
        cbfShortcut.rollouts, cbfShortcut.maxArrivalGap);

    writePath(output, cbfRepresentative.motion);
    const std::string omplOutput = baselinePathFor(output);
    writePath(omplOutput, omplRepresentative.motion);
    std::printf("CBF trajectory: %s\nOMPL trajectory: %s\n", output.c_str(), omplOutput.c_str());
    return 0;
}
