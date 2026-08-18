#define BOOST_TEST_MODULE SDFTest
#include <boost/test/unit_test.hpp>

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/sdf/GridSDF.h>

namespace sdf = ompl::sdf;

namespace
{
    // Two spheres stand in for "an environment": GridSDF only ever sees a
    // signed-distance function, never the geometry. Distance to the scene is the
    // min over the two spheres.
    auto sceneDistance() -> sdf::DistanceFn
    {
        return [](const Eigen::Vector3d &p)
        {
            const double d0 = (p - Eigen::Vector3d(0.5, 0.0, 0.0)).norm() - 0.2;
            const double d1 = (p - Eigen::Vector3d(-0.4, 0.3, 0.0)).norm() - 0.15;
            return std::min(d0, d1);
        };
    }

    auto makeGrid() -> sdf::GridSDF
    {
        const Eigen::AlignedBox3d bounds(Eigen::Vector3d(-1.0, -1.0, -1.0), Eigen::Vector3d(1.0, 1.0, 1.0));
        return sdf::GridSDF(sceneDistance(), bounds, /*voxel=*/0.05);
    }
}  // namespace

BOOST_AUTO_TEST_CASE(GridMatchesSourceWithinDiscretization)
{
    const auto grid = makeGrid();
    const auto exact = sceneDistance();

    // Cached value tracks the source up to discretization error.
    const Eigen::Vector3d p(0.85, 0.1, 0.0);
    BOOST_CHECK_SMALL(grid.distance(p) - exact(p), 0.01);

    // Negative inside an obstacle, positive outside.
    BOOST_CHECK_LT(grid.distance(Eigen::Vector3d(0.5, 0.0, 0.0)), 0.0);
    BOOST_CHECK_GT(grid.distance(Eigen::Vector3d(0.0, 0.0, 0.5)), 0.0);
}

BOOST_AUTO_TEST_CASE(GradientPointsAwayFromObstacle)
{
    const auto grid = makeGrid();

    // Just outside the first sphere on its +x side: gradient ~ +x, unit norm.
    const Eigen::Vector3d g = grid.gradient(Eigen::Vector3d(0.85, 0.0, 0.0));
    BOOST_CHECK_CLOSE(g.norm(), 1.0, 5.0);
    BOOST_CHECK_GT(g.normalized().dot(Eigen::Vector3d::UnitX()), 0.9);
}

BOOST_AUTO_TEST_CASE(GradientIsDerivativeOfInterpolatedValue)
{
    const auto grid = makeGrid();
    // Deliberately away from a voxel boundary, where the trilinear field is smooth.
    const Eigen::Vector3d p(0.837, 0.113, 0.071);
    const Eigen::Vector3d analytic = grid.gradient(p);
    Eigen::Vector3d finiteDifference;
    constexpr double epsilon = 1e-6;
    for (int d = 0; d < 3; ++d)
    {
        Eigen::Vector3d plus = p;
        Eigen::Vector3d minus = p;
        plus[d] += epsilon;
        minus[d] -= epsilon;
        finiteDifference[d] = (grid.distance(plus) - grid.distance(minus)) / (2.0 * epsilon);
    }

    BOOST_CHECK_LE((analytic - finiteDifference).norm(), 1e-9);
}

BOOST_AUTO_TEST_CASE(BoundsAndDimensions)
{
    const auto grid = makeGrid();

    BOOST_CHECK(grid.inBounds(Eigen::Vector3d(0.0, 0.0, 0.0)));
    BOOST_CHECK(not grid.inBounds(Eigen::Vector3d(2.0, 0.0, 0.0)));

    // 2.0 extent / 0.05 voxel -> 41 nodes per axis.
    BOOST_CHECK_EQUAL(grid.dimensions().x(), 41);
}

BOOST_AUTO_TEST_CASE(BatchedQueriesMatchScalarQueries)
{
    const auto grid = makeGrid();
    Eigen::Matrix<double, 3, 5> points;
    points << 0.837, -0.42, 0.0, -1.2, 1.2,
              0.113, 0.31, -0.7, 0.4, -1.1,
              0.071, -0.08, 0.9, 1.3, -0.2;

    Eigen::Matrix<double, 5, 1> distances;
    Eigen::Matrix<double, 3, 5> gradients;
    grid.valueGradientBatch(points, distances, gradients);

    for (Eigen::Index i = 0; i < points.cols(); ++i)
    {
        const sdf::ValueGradient scalar = grid.valueAndGradient(points.col(i));
        BOOST_CHECK_EQUAL(distances[i], scalar.value);
        BOOST_CHECK_SMALL((gradients.col(i) - scalar.gradient).norm(), 1e-15);
    }

    Eigen::Matrix<double, 5, 1> distanceOnly;
    grid.distanceBatch(points, distanceOnly);
    BOOST_CHECK_SMALL((distanceOnly - distances).norm(), 1e-15);
}

BOOST_AUTO_TEST_CASE(OutsideQueriesClampAndHaveNoOutwardDerivative)
{
    const auto grid = makeGrid();
    const Eigen::Vector3d outside(1.4, 0.113, 0.071);
    const Eigen::Vector3d boundary(1.0, outside.y(), outside.z());
    const sdf::ValueGradient clamped = grid.valueAndGradient(outside);

    BOOST_CHECK_EQUAL(clamped.value, grid.distance(boundary));
    BOOST_CHECK_EQUAL(clamped.gradient.x(), 0.0);
}
