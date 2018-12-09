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


#ifndef SCAN_MATCHER_LIB_OCCUPIED_SPACE_COST_FUNCTION_2D_H
#define SCAN_MATCHER_LIB_OCCUPIED_SPACE_COST_FUNCTION_2D_H

#include <pybind11/numpy.h>
#include <ceres/ceres.h>

namespace py = pybind11;

namespace cartographer {
namespace mapping {
namespace scan_matching {

// Creates a cost function for matching the 'point_cloud' to the 'grid' with
// a 'pose'. The cost increases with poorer correspondence of the grid and the
// point observation (e.g. points falling into less occupied space).
ceres::CostFunction* CreateOccupiedSpaceCostFunction2D(
  const double scaling_factor, const py::array_t<double >& point_cloud, const py::array_t<double >& grid);

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif //SCAN_MATCHER_LIB_OCCUPIED_SPACE_COST_FUNCTION_2D_H
