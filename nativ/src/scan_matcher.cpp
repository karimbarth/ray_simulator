#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>

#include <occupied_space_cost_function_2d.h>

namespace py = pybind11;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor
{
  template<typename T>
  bool operator()( const T *const x, T *residual ) const
  {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

py::tuple match(py::array_t<double> point_cloud ,py::array_t<double> grid_map, double resolution, py::array_t<double> initial_pose )
{
  //auto grid_map_proxy = grid_map.unchecked<2>();
  //auto point_cloud_proxy = point_cloud.unchecked<2>();

  double ceres_pose_estimate[3] = {initial_pose.at(0), initial_pose.at(1), initial_pose.at(2)}; // initial pose

  Problem problem;
  auto cost_function = cartographer::mapping::scan_matching::CreateOccupiedSpaceCostFunction2D(resolution, point_cloud, grid_map);
  problem.AddResidualBlock(cost_function, nullptr, ceres_pose_estimate);
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;

  Solve( options, &problem, &summary );
  std::cout << summary.BriefReport() << "\n";
  /*
  double x = 0.5;
  Problem problem;
  CostFunction *cost_function = new AutoDiffCostFunction<CostFunctor, 1, 1>( new CostFunctor );
  problem.AddResidualBlock( cost_function, NULL, &x );

  Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  Solve( options, &problem, &summary );
  //std::cout << summary.BriefReport() << "\n";
  */


  return py::make_tuple(ceres_pose_estimate[0], ceres_pose_estimate[1], ceres_pose_estimate[2]);
}

//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>

//using namespace boost::python;

PYBIND11_MODULE( scan_matcher, m )
{
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def( "match", &match, "Match the given point cloud with the given grid map, returns the relative translation and orientation" );
}
