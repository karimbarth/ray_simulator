#include <iostream>
#include <pybind11/pybind11.h>
#include <ceres/ceres.h>
namespace py = pybind11;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor {
    template <typename T> bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

int add(int i, int j) {

    double x = 0.5;
    const double initial_x = x;

    Problem problem;
    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, NULL, &x);

    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    return i + j + 1;
}

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;

PYBIND11_MODULE(scan_matcher, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
