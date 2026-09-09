// Paired UR5 adaptation of the QP-free CBF gate from LQR-CBF-RRT* versus
// OMPL's CBF-QP plus Lipschitz certificates.  Reuse the MBM parser, exact
// primitive distances, UR5 audit, and result schema from OMPL's benchmark.
#define main embedded_ur5_mbm_main
#include "UR5MBMBenchmark.cpp"
#undef main

#include <ompl/base/PlannerData.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/util/Console.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>

namespace
{
    class QPFreeUR5Gate final : public ompl::cbf::ControlFilter
    {
    public:
        QPFreeUR5Gate(const Barrier &barrier, const Filter::Parameters &parameters)
          : barrier_(barrier), parameters_(parameters),
            decreaseRates_(barrier.decreaseRates(parameters.maxSpeed))
        {
            threshold_.setZero();
        }

        Status filter(const Configuration &q, const Control &nominal, double duration,
                      Control &applied) const override
        {
            ++calls_;
            applied.setZero();
            if (!(duration > 0.0))
                return reject();

            Control lower = -parameters_.maxSpeed.cwiseAbs();
            Control upper = parameters_.maxSpeed.cwiseAbs();
            if (parameters_.respectJointLimits)
            {
                lower = lower.cwiseMax((UR5::lowerBounds() - q) / duration);
                upper = upper.cwiseMin((UR5::upperBounds() - q) / duration);
            }
            applied = nominal.cwiseMax(lower).cwiseMin(upper);

            // Upstream selects nearby obstacles before evaluating its gate.  Use the
            // UR5 barrier's sound counterpart: a row beyond this threshold is satisfied
            // by every control in the shared control box and cannot change accept/reject.
            const double horizon = parameters_.kappa > 0.0
                                       ? std::max(duration, 1.0 / parameters_.kappa)
                                       : std::numeric_limits<double>::infinity();
            if (horizon != cachedHorizon_)
            {
                threshold_ = decreaseRates_ * horizon;
                cachedHorizon_ = horizon;
            }
            barrier_.evaluateScreened(q, threshold_, evaluation_);
            rows_ += static_cast<std::size_t>(evaluation_.active);
            if (!evaluation_.inBounds)
            {
                applied.setZero();
                return reject();
            }

            for (Eigen::Index row = 0; row < evaluation_.active; ++row)
            {
                const int constraint = evaluation_.constraint[row];
                const double h = evaluation_.values[constraint];
                if (h < 0.0 ||
                    evaluation_.rows.row(row).dot(applied) + parameters_.kappa * h < 0.0)
                {
                    applied.setZero();
                    return reject();
                }
            }

            return applied.isApprox(nominal, 0.0) ? Status::Unchanged : Status::Filtered;
        }

        const char *name() const override
        {
            return "ur5-qp-free-cbf-gate";
        }

        std::size_t calls() const
        {
            return calls_;
        }

        std::size_t rejected() const
        {
            return rejected_;
        }

        double meanRows() const
        {
            return calls_ ? static_cast<double>(rows_) / static_cast<double>(calls_) : 0.0;
        }

    private:
        Status reject() const
        {
            ++rejected_;
            return Status::Blocked;
        }

        const Barrier &barrier_;
        Filter::Parameters parameters_;
        Barrier::Values decreaseRates_;
        mutable Barrier::Values threshold_;
        mutable Barrier::Evaluation evaluation_;
        mutable double cachedHorizon_{-1.0};
        mutable std::size_t calls_{0};
        mutable std::size_t rejected_{0};
        mutable std::size_t rows_{0};
    };

    class SeededUR5Sampler final : public ob::RealVectorStateSampler
    {
    public:
        SeededUR5Sampler(const ob::StateSpace *space, std::uint_fast32_t seed)
          : ob::RealVectorStateSampler(space)
        {
            rng_.setLocalSeed(seed);
        }
    };

    class SeededUR5RRT final : public og::RRT
    {
    public:
        SeededUR5RRT(const ob::SpaceInformationPtr &si, std::uint_fast32_t seed) : og::RRT(si)
        {
            rng_.setLocalSeed(seed);
        }
    };

    struct PairedResult
    {
        Result result;
        std::size_t rejected{0};
        double meanRows{0.0};
    };

    template <typename ControlFilter>
    PairedResult runPlanner(const Problem &problem, const Barrier &audited,
                            ControlFilter &filter, double stepSize, double range,
                            unsigned int iterations, std::uint_fast32_t seed,
                            bool connect)
    {
        auto space = std::make_shared<Space>(filter, stepSize, UR5::velocityLimits());
        space->setBounds(jointBounds());
        space->setMaxStepScale(std::numeric_limits<double>::infinity());
        space->setSafeHops(true);
        Space::EarlyTermination early;
        early.enabled = true;
        early.maxFilterCalls = 40;
        space->setEarlyTermination(early);
        space->setStateSamplerAllocator(
            [seed](const ob::StateSpace *stateSpace)
            {
                return std::make_shared<SeededUR5Sampler>(stateSpace, seed);
            });

        auto si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<ob::AllValidStateValidityChecker>(si));
        si->setMotionValidator(std::make_shared<ompl::cbf::FilteredMotionValidator>(si));
        si->setup();

        ob::ScopedState<> start(space), goal(space);
        for (int joint = 0; joint < dimension; ++joint)
        {
            start[joint] = problem.start[joint];
            goal[joint] = problem.goal[joint];
        }
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(start, goal, 0.1);

        ob::PlannerPtr planner;
        if (connect)
        {
            auto rrtConnect = std::make_shared<og::RRTConnect>(si);
            rrtConnect->setRange(range);
            // Directed steering often advances safely without reaching the random
            // target. Retain that useful prefix equally for both filters.
            rrtConnect->setRetainPartialSteering(true);
            planner = rrtConnect;
        }
        else
        {
            auto rrt = std::make_shared<SeededUR5RRT>(si, seed + 1u);
            rrt->setRange(range);
            rrt->setGoalBias(0.05);
            planner = rrt;
        }
        planner->setProblemDefinition(pdef);
        planner->setup();

        PairedResult output;
        const auto begin = std::chrono::steady_clock::now();
        const ob::PlannerStatus status =
            planner->solve(ob::IterationTerminationCondition(iterations));
        output.result.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        output.result.solved = status == ob::PlannerStatus::EXACT_SOLUTION;
        output.result.evaluations = space->statistics().steps;
        if (space->statistics().steps > 0)
        {
            const double calls = static_cast<double>(space->statistics().steps);
            output.result.radPerCall = space->statistics().travel / calls;
            output.result.coarse = static_cast<double>(space->statistics().coarse) / calls;
        }

        ob::PlannerData data(si);
        planner->getPlannerData(data);
        output.result.vertices = data.numVertices();

        if (output.result.solved)
        {
            const auto solution =
                std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
            audit(*solution, problem, audited, output.result, nullptr);
        }

        if constexpr (std::is_same_v<ControlFilter, QPFreeUR5Gate>)
        {
            output.rejected = filter.rejected();
            output.meanRows = filter.meanRows();
        }
        return output;
    }

    struct Summary
    {
        std::size_t attempted{0};
        std::size_t solved{0};
        std::size_t unsafe{0};
        std::size_t selfColliding{0};
        std::size_t misses{0};
        std::vector<double> milliseconds;
        std::vector<double> calls;
        std::vector<double> vertices;
        std::vector<double> paths;
    };

    void add(Summary &summary, const PairedResult &run)
    {
        ++summary.attempted;
        summary.solved += run.result.solved ? 1u : 0u;
        summary.unsafe += run.result.unsafeStates;
        summary.selfColliding += run.result.selfColliding;
        summary.misses += run.result.misses;
        summary.milliseconds.push_back(1e3 * run.result.seconds);
        summary.calls.push_back(static_cast<double>(run.result.evaluations));
        summary.vertices.push_back(static_cast<double>(run.result.vertices));
        if (run.result.solved)
            summary.paths.push_back(run.result.pathLength);
    }

    void printSummary(const char *method, const Summary &summary)
    {
        std::cout << method << ',' << summary.solved << '/' << summary.attempted << ','
                  << median(summary.milliseconds) << ',' << median(summary.calls) << ','
                  << median(summary.vertices) << ',' << median(summary.paths) << ','
                  << summary.unsafe << ',' << summary.selfColliding << ',' << summary.misses
                  << '\n';
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
        {
            std::cerr << "usage: " << argv[0]
                      << " scenes.txt [perScene=3] [iterations=2000] [voxel=.03]"
                         " [step=.05] [range=2] [margin=.005] [buffer=.01]"
                         " [kappa=8] [seed=17] [csv] [planner=rrt] [scene=all]"
                         " [problem_ids=all]\n";
            return EXIT_FAILURE;
        }

        const std::string scenesPath = argv[1];
        const int perScene = argc > 2 ? std::stoi(argv[2]) : 3;
        const unsigned int iterations = argc > 3 ? std::stoul(argv[3]) : 2000u;
        const double voxel = argc > 4 ? std::stod(argv[4]) : 0.03;
        const double step = argc > 5 ? std::stod(argv[5]) : 0.05;
        const double range = argc > 6 ? std::stod(argv[6]) : 2.0;
        const double margin = argc > 7 ? std::stod(argv[7]) : 0.005;
        const double buffer = argc > 8 ? std::stod(argv[8]) : 0.01;
        const double kappa = argc > 9 ? std::stod(argv[9]) : 8.0;
        const std::uint_fast32_t firstSeed = argc > 10 ? std::stoul(argv[10]) : 17u;
        const std::string csvPath = argc > 11 ? argv[11] : std::string();
        const std::string plannerName = argc > 12 ? argv[12] : "rrt";
        const bool connect = plannerName == "rrtconnect";
        const std::string sceneFilter = argc > 13 ? argv[13] : "all";
        const std::string problemIds = argc > 14 ? argv[14] : "all";
        std::set<int> problemAllowlist;
        if (problemIds != "all")
        {
            std::istringstream ids(problemIds);
            std::string id;
            while (std::getline(ids, id, ','))
                problemAllowlist.insert(std::stoi(id));
        }
        if (perScene <= 0 || iterations == 0 || voxel <= 0.0 || step <= 0.0 || range <= 0.0)
            throw std::invalid_argument("counts and geometric scales must be positive");
        if (!connect && plannerName != "rrt")
            throw std::invalid_argument("planner must be rrt or rrtconnect");

        ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
        const std::vector<Problem> problems = readProblems(scenesPath);
        const UR5 robot;
        Filter::Parameters parameters;
        parameters.kappa = kappa;
        parameters.maxSpeed = UR5::velocityLimits();
        parameters.respectJointLimits = true;

        std::ofstream csv;
        if (!csvPath.empty())
        {
            csv.open(csvPath);
            if (!csv)
                throw std::runtime_error("cannot write " + csvPath);
            csv << "seed,scene,problem,method,planner,eligible,solved,seconds,filter_calls,vertices,"
                   "path_length,audited_states,unsafe_states,self_colliding,replay_misses,"
                   "rad_per_call,coarse_fraction,rejected,mean_rows\n";
        }

        Summary qpSummary, gateSummary;
        std::map<std::string, int> seen;
        std::size_t eligible = 0;
        std::size_t skipped = 0;
        for (const Problem &problem : problems)
        {
            if (sceneFilter != "all" && problem.scene != sceneFilter)
                continue;
            if (!problemAllowlist.empty() && problemAllowlist.count(problem.index) == 0)
                continue;
            if (seen[problem.scene]++ >= perScene)
                continue;

            const sdf::GridSDF field(problem.field(), UR5::reachableBounds(), voxel);
            const Barrier audited(robot, field, margin, Barrier::defaultSelfMargin);
            const Barrier guard = Barrier::guarding(robot, field, margin, buffer,
                                                     Barrier::defaultSelfMargin);
            if (!audited.isSafe(problem.start) || !audited.isSafe(problem.goal))
            {
                ++skipped;
                continue;
            }

            const std::uint_fast32_t runSeed =
                firstSeed + static_cast<std::uint_fast32_t>(eligible * 2u);
            PairedResult qp;
            PairedResult gate;
            auto runQP = [&]
            {
                Filter filter(guard, parameters);
                qp = runPlanner(problem, audited, filter, step, range, iterations, runSeed,
                                connect);
            };
            auto runGate = [&]
            {
                QPFreeUR5Gate filter(guard, parameters);
                gate = runPlanner(problem, audited, filter, step, range, iterations, runSeed,
                                  connect);
            };
            if ((eligible & 1u) == 0u)
            {
                runQP();
                runGate();
            }
            else
            {
                runGate();
                runQP();
            }

            add(qpSummary, qp);
            add(gateSummary, gate);
            const auto write = [&](const char *method, const PairedResult &run)
            {
                if (!csv)
                    return;
                const Result &r = run.result;
                csv << runSeed << ',' << problem.scene << ',' << problem.index << ',' << method
                    << ',' << plannerName << ",1," << (r.solved ? 1 : 0) << ','
                    << std::setprecision(12) << r.seconds
                    << ',' << r.evaluations << ',' << r.vertices << ',' << r.pathLength << ','
                    << r.auditedStates << ',' << r.unsafeStates << ',' << r.selfColliding << ','
                    << r.misses << ',' << r.radPerCall << ',' << r.coarse << ',' << run.rejected
                    << ',' << run.meanRows << '\n';
            };
            write("qp_lipschitz", qp);
            write("qp_free_gate", gate);
            ++eligible;
        }

        std::cout << "UR5 MBM paired " << plannerName << " benchmark: " << eligible
                  << " eligible, " << skipped << " endpoint-ineligible; " << iterations
                  << " iterations/run\n";
        std::cout << "method,solved/attempted,median_ms,median_filter_calls,median_vertices,"
                     "median_path_length,unsafe_states,self_colliding,replay_misses\n";
        printSummary("qp_lipschitz", qpSummary);
        printSummary("qp_free_gate", gateSummary);
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
