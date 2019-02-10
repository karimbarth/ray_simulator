#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <ceres/ceres.h>
#include <stdio.h>
#include <omp.h>
#include <cmath>
#include <random>

#include <trng/yarn2.hpp>
#include <trng/uniform_dist.hpp>

#include <occupied_space_cost_function_2d.h>

namespace py = pybind11;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


double evaluate_cost_function( py::array_t<double> point_cloud, py::array_t<double> grid_map, double resolution,
                               py::array_t<double> position )
{
  Problem problem;
  double eval_position[3] = { position.at( 0 ), position.at( 1 ), position.at( 2 ) };
  auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D( resolution, point_cloud,
                                                                                                grid_map );
  problem.AddResidualBlock( cost_function, nullptr, eval_position );
  double cost = 0.0;
  problem.Evaluate( Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr );

  return cost;
}


py::tuple match( py::array_t<double> point_cloud, py::array_t<double> grid_map, double resolution,
                 py::array_t<double> initial_pose, bool output )
{

  double ceres_pose_estimate[3] = { initial_pose.at( 0 ), initial_pose.at( 1 ), initial_pose.at( 2 ) }; // initial pose

  Problem problem;
  auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D( resolution, point_cloud,
                                                                                                grid_map );
  problem.AddResidualBlock( cost_function, nullptr, ceres_pose_estimate );
  Solver::Options options;
  options.minimizer_progress_to_stdout = output;
  Solver::Summary summary;

  Solve( options, &problem, &summary );
  if ( output )
    std::cout << summary.BriefReport() << "\n";

  return py::make_tuple( ceres_pose_estimate[0], ceres_pose_estimate[1], ceres_pose_estimate[2] );
}


Eigen::MatrixXd
evaluate_position_with_samples( py::array_t<double> point_cloud, py::array_t<double> grid_map, double resolution,
                                py::array_t<double> sensor_position, int sample_count, double evaluation_radius )
{
  // use random generator, which is capable of using in parallel environments 
  trng::yarn2 gen;

  Eigen::MatrixXd result(5, sample_count);

  #pragma omp parallel for
  for ( int n = 0; n < sample_count; ++n )
  {
    int size = omp_get_num_threads();
    int rank= omp_get_thread_num();
    gen.split(size, rank);
    trng::uniform_dist<> angle_distribution(0.0,2*M_PI);
    trng::uniform_dist<> distance_distribution(0.0, evaluation_radius * resolution);

    double angle = angle_distribution(gen);
    double length = distance_distribution(gen);

    double x = std::cos(angle) * length + sensor_position.at( 0 );
    double y = std::sin(angle) * length + sensor_position.at( 1 );

    double ceres_pose_estimate[3] = { x, y, sensor_position.at( 2 ) };
    Problem problem;
    auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D( resolution,
                                                                                                  point_cloud,
                                                                                                  grid_map );
    problem.AddResidualBlock( cost_function, nullptr, ceres_pose_estimate );
    Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

    Solve( options, &problem, &summary );

    Eigen::Vector2d start_translation_error;
    start_translation_error << sensor_position.at( 0 ) - x, sensor_position.at( 1 ) - y;
    Eigen::Vector2d end_translation_error;
    end_translation_error << (sensor_position.at( 0 ) - ceres_pose_estimate[0]), (sensor_position.at( 1 ) - ceres_pose_estimate[1]);

    result.col(n) << start_translation_error.norm(), end_translation_error.norm(), 0,
                     std::fabs(sensor_position.at( 2 ) - ceres_pose_estimate[2]), summary.iterations.size();
  }

  return result;
}


PYBIND11_MODULE( scan_matcher, m )
{
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def( "match", &match,
         "Match the given point cloud with the given grid map, returns the relative translation and orientation" );
  m.def( "evaluate_cost_function", &evaluate_cost_function, "Evaluate the cost function at a given position" );
  m.def( "evaluate_position_with_samples", &evaluate_position_with_samples,
         "Perform scan matching around given position" );
}
