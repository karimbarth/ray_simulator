#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <ceres/ceres.h>

#include <occupied_space_cost_function_2d.h>

namespace py = pybind11;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


double evaluate_cost_function(py::array_t<double> point_cloud ,py::array_t<double> grid_map, double resolution, py::array_t<double> position)
{
  Problem problem;
  double eval_position[3] = {position.at(0), position.at(1), position.at(2)};
  auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D(resolution, point_cloud, grid_map);
  problem.AddResidualBlock(cost_function, nullptr, eval_position);
  double cost = 0.0;
  problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);

  return cost;
}


py::tuple match(py::array_t<double> point_cloud ,py::array_t<double> grid_map, double resolution, py::array_t<double> initial_pose, bool output)
{

  double ceres_pose_estimate[3] = {initial_pose.at(0), initial_pose.at(1), initial_pose.at(2)}; // initial pose

  Problem problem;
  auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D(resolution, point_cloud, grid_map);
  problem.AddResidualBlock(cost_function, nullptr, ceres_pose_estimate);
  Solver::Options options;
  options.minimizer_progress_to_stdout = output;
  Solver::Summary summary;

  Solve( options, &problem, &summary );
  if(output)
    std::cout << summary.BriefReport() << "\n";

  return py::make_tuple(ceres_pose_estimate[0], ceres_pose_estimate[1], ceres_pose_estimate[2]);
}


PYBIND11_MODULE( scan_matcher, m )
{
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def( "match", &match, "Match the given point cloud with the given grid map, returns the relative translation and orientation" );
  m.def( "evaluate_cost_function", &evaluate_cost_function, "Evaluate the cost function at a given position" );
}
