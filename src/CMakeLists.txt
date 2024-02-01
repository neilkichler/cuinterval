cmake_minimum_required (VERSION 3.24)

set(LIB_PATH "${PROJECT_SOURCE_DIR}/include/cuinterval")

set(headers 
    "${LIB_PATH}/arithmetic/intrinsic.cuh"
    "${LIB_PATH}/arithmetic/basic.cuh"
    "${LIB_PATH}/cuinterval.h"
)

add_library(cuinterval "main.cu" ${headers})

target_include_directories(cuinterval PUBLIC "${PROJECT_SOURCE_DIR}/include")

# C++17 is required for all users of the library
target_compile_features(cuinterval PRIVATE cxx_std_17)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${headers})

# set_target_properties(CudaIntervals PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

