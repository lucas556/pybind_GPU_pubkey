
cmake_minimum_required(VERSION 3.18)
project(pygpupub LANGUAGES C CXX CUDA)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Include headers
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

# CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    pybind_module_batch.cu
)

# Create shared library
pybind11_add_module(pygpupub MODULE ${SOURCES})
target_link_libraries(pygpupub PRIVATE CUDA::cudart)
