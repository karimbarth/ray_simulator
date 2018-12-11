/*
 * Copyright 2018 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified by Karim Barth
 */


#include "occupied_space_cost_function_2d.h"
#include "ceres/cubic_interpolation.h"

namespace cartographer
{
namespace mapping
{
namespace scan_matching
{
namespace
{

class OccupiedSpaceCostFunction2D
{
public:
  OccupiedSpaceCostFunction2D(const double resolution,
                              const py::array_t<double >& point_cloud,
                              const py::array_t<double >& grid)
    : resolution_(resolution),
    point_cloud_(point_cloud),
    grid_(grid) {}

  template <typename T>
  bool operator()( const T* const pose, T* residual) const {

    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);

    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const GridArrayAdapter adapter(grid_);
    ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);

    for(size_t i = 0; i < point_cloud_.shape(1); ++i)
    {
      const Eigen::Matrix<T, 3, 1> point((T(point_cloud_.at(0,i))), (T(point_cloud_.at(1,i))), T(1.));
      const Eigen::Matrix<T, 3, 1> world = transform * point;

      T grid_map_value;
      interpolator.Evaluate(world[0] * resolution_, world[1] * resolution_ , &grid_map_value);
      residual[i] = T(1.) - grid_map_value;
    }

    return true;

  }

private:

  class GridArrayAdapter {
  public:
    enum { DATA_DIMENSION = 1 };

    explicit GridArrayAdapter(const py::array_t<double>& grid)
    : grid_(grid) {}

    void GetValue(const int row, const int column, double* const value) const {
      // inside the grid
      if(row >= 0 && row < grid_.shape(0) && column >= 0 && column  < grid_.shape(1))
        *value = grid_.at(row, column);
      else
        *value = -10;
    }

  private:
    const py::array_t<double>& grid_;
  };

  OccupiedSpaceCostFunction2D(const OccupiedSpaceCostFunction2D&) = delete;
  OccupiedSpaceCostFunction2D& operator=(const OccupiedSpaceCostFunction2D&) = delete;

  const double resolution_;
  const py::array_t<double>& point_cloud_;
  const py::array_t<double>& grid_;

};

} // namespace

ceres::CostFunction* CreateOccupiedSpaceCostFunction2D(
  const double resolution, const py::array_t<double>& point_cloud,
  const py::array_t<double>& grid) {
  return new ceres::AutoDiffCostFunction<OccupiedSpaceCostFunction2D,
    ceres::DYNAMIC /* residuals */,
    3 /* pose variables */>(
    new OccupiedSpaceCostFunction2D(resolution, point_cloud, grid),
    point_cloud.shape(1));
}

} // scan_matching
} // mapping
} // cartographer

