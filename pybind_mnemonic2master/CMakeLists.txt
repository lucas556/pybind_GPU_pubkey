cmake_minimum_required(VERSION 3.18)
project(pybind_mnemonic2master LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# 包含自定义 CUDA HMAC/PBKDF2 路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(pybind_mnemonic2master MODULE pybind_mnemonic2master.cu)
set_target_properties(pybind_mnemonic2master PROPERTIES PREFIX "")

# 适配 2080, 3080, 4080, 5080
set_target_properties(pybind_mnemonic2master PROPERTIES CUDA_ARCHITECTURES "75;86;89;90")

# 允许 relaxed constexpr（可选）
target_compile_options(pybind_mnemonic2master PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(pybind_mnemonic2master PRIVATE pybind11::module)
