cmake_minimum_required (VERSION 3.5)
project(scan_matcher_lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Ceres REQUIRED)
#find_package(TRNG REQUIRED)
find_library(TRNG_LIBRARIES NAMES libtrng4.so PATHS /usr/local/lib/)

include_directories(
	${CERES_INCLUDE_DIRS}
    ${TRNG_INCLUDE_DIRS}
	include	
)

set(SOURCES
 src/scan_matcher.cpp
 src/occupied_space_cost_function_2d.cpp
)

find_package(pybind11 REQUIRED)

pybind11_add_module(scan_matcher ${SOURCES})
target_link_libraries(scan_matcher PRIVATE ${CERES_LIBRARIES})

target_link_libraries(scan_matcher PRIVATE ${TRNG_LIBRARIES})

install(TARGETS scan_matcher DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)

