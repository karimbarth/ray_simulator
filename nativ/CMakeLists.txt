cmake_minimum_required (VERSION 3.5)
project(scan_matcher_lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Ceres REQUIRED)

include_directories(
	${CERES_INCLUDE_DIRS}
	include	
)

set(SOURCES
 src/scan_matcher.cpp
 src/occupied_space_cost_function_2d.cpp
)

find_package(pybind11 REQUIRED)

pybind11_add_module(scan_matcher ${SOURCES})
target_link_libraries(scan_matcher PRIVATE ${CERES_LIBRARIES})
