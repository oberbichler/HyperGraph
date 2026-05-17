cmake_minimum_required(VERSION 3.24)

project(HyperGraph-py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---

include(FetchContent)

find_package(pybind11 CONFIG REQUIRED)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_Populate(eigen)

add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")

find_package(BLAS QUIET)

# --- Target ---

pybind11_add_module(hypergraph src/Module.cpp)

target_include_directories(hypergraph PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/external_libraries"
)

target_link_libraries(hypergraph PRIVATE Eigen3::Eigen)

target_compile_definitions(hypergraph PRIVATE
    EIGEN_DEFAULT_TO_ROW_MAJOR
    HYPERGRAPH_VERSION="${SKBUILD_PROJECT_VERSION}"
    HYPERGRAPH_EXCEPTIONS
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(hypergraph PRIVATE -fsized-deallocation)
endif()

if(BLAS_FOUND)
    target_compile_definitions(hypergraph PRIVATE EIGEN_USE_BLAS)
    target_link_libraries(hypergraph PRIVATE ${BLAS_LIBRARIES})
endif()

install(TARGETS hypergraph DESTINATION .)
