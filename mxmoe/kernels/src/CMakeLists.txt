cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architecture for the standalone benchmark")
add_compile_definitions(COMPUTE_ARCH=${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 128 -diag-suppress 2417, -diag-suppress 20050")

project(mxmoe LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#################################################################
# Config
#################################################################
set(cutlass_dir "${CMAKE_SOURCE_DIR}/3rdparty/cutlass")
set(cccl_dir "${CMAKE_SOURCE_DIR}/3rdparty/cccl")

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${cutlass_dir}/include
  ${cutlass_dir}/tools/util/include
)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${cccl_dir}/lib/cmake/cccl")


#################################################################
# Targets
#################################################################

set(GENERATED_DIR "${CMAKE_SOURCE_DIR}/generated" CACHE PATH "Generated grouped GEMM sources")
include_directories(${GENERATED_DIR})

file(GLOB GENERATED_CU_FILES "${GENERATED_DIR}/*.cu")

add_library(gen_lib SHARED ${GENERATED_CU_FILES})
target_compile_options(gen_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo --use_fast_math>) 
target_link_libraries(gen_lib
  PRIVATE
  fmt::fmt
)

add_executable(test test.cu)
target_link_options(test PRIVATE "-Wl,--no-as-needed")
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo --use_fast_math>) 
target_link_libraries(test
  PRIVATE
  fmt::fmt
  CCCL::Thrust
  nlohmann_json::nlohmann_json
  argparse
  gen_lib
)

# Extension
# python
set(Python3_ROOT_DIR "${CONDA_PREFIX}")
find_package(Python REQUIRED COMPONENTS Development Interpreter)
set(DEV_MODULE Development.Module)

# set(Python3_ROOT_DIR "${CONDA_PREFIX}")
# find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
# if(Python3_FOUND)
#     message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
#     message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
# endif()

# torch
execute_process(
  COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_cmake_path)
list(APPEND CMAKE_PREFIX_PATH ${torch_cmake_path})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/pybind11 pybind11)

# mxmoe_ops (pybind) needs act_kernel.cuh which is not in-tree; L2 gate only needs `test`.
# Keep the target optional so default `cmake --build build` still builds gen_lib + test on A100.
if(EXISTS "${CMAKE_SOURCE_DIR}/include/act_kernel.cuh")
  pybind11_add_module(mxmoe_ops ref_bind.cu)
  target_compile_options(mxmoe_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo --use_fast_math>)
  target_link_libraries(mxmoe_ops PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} fmt::fmt Python::Python)
  target_include_directories(mxmoe_ops PRIVATE
      ${TORCH_INCLUDE_DIRS}
      ${Python3_INCLUDE_DIRS}
  )
else()
  message(WARNING "act_kernel.cuh missing — skipping mxmoe_ops pybind module (bench `test` target still builds)")
endif()

#################################################################
# 3rdparty
#################################################################
# fmt
set(BUILD_SHARED_LIBS ON)
add_subdirectory(3rdparty/fmt)

# CCCL
find_package(CCCL REQUIRED CONFIG COMPONENTS Thrust CUB)

# json
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# argparse
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)