cmake_minimum_required(VERSION 3.18)
project(gpu_pbkdf2 LANGUAGES CXX CUDA)

# 下载并使用 pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# 查找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置为 GPU 模块构建（必需）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 源文件（必须使用 .cu 以支持 <<< >>> 和 threadIdx）
set(SOURCES
    pybind_module_batch.cu
)

# 添加 Python 模块（.cu 文件将自动使用 nvcc 编译）
pybind11_add_module(gpu_pbkdf2 ${SOURCES})

# 包含 CUDA 头文件
target_include_directories(gpu_pbkdf2 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# 链接 CUDA runtime
target_link_libraries(gpu_pbkdf2 PRIVATE CUDA::cudart)
