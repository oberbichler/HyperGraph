cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(HyperGraph-py)

find_package(BLAS)

include_directories(
    "include"
    "external_libraries"
)

add_subdirectory(external_libraries/pybind11)

add_definitions(
    -DEIGEN_DEFAULT_TO_ROW_MAJOR
    -DHYPERGRAPH_VERSION="${HYPERGRAPH_VERSION}"
    -DHYPERGRAPH_EXCEPTIONS
)

pybind11_add_module(hypergraph src/Module.cpp)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(hypergraph PUBLIC -fsized-deallocation)
endif()

if(BLAS_FOUND)
    add_definitions(-DEIGEN_USE_BLAS)
    target_link_libraries(hypergraph PRIVATE ${BLAS_LIBRARIES})
endif()

install(TARGETS hypergraph DESTINATION bin)