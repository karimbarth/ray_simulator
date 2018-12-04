cmake_minimum_required (VERSION 3.5)
project(scan_matcher_lib)

find_package(Ceres REQUIRED)

include_directories(
	${CERES_INCLUDE_DIRS}
)

find_package(pybind11 REQUIRED)

pybind11_add_module(scan_matcher scan_matcher.cpp)
target_link_libraries(scan_matcher PRIVATE ${CERES_LIBRARIES})
