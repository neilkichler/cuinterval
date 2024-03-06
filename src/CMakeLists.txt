cmake_minimum_required (VERSION 3.24)

set(INCLUDE_PATH "${cuinterval_SOURCE_DIR}/include")
set(LIB_PATH "${INCLUDE_PATH}/cuinterval")

set(headers 
    "${LIB_PATH}/arithmetic/intrinsic.cuh"
    "${LIB_PATH}/arithmetic/basic.cuh"
    "${LIB_PATH}/cuinterval.h"
)

add_library(cuinterval "main.cu" ${headers})

target_include_directories(cuinterval PUBLIC "${INCLUDE_PATH}")

# C++17 is required for all users of the library
target_compile_features(cuinterval PUBLIC cxx_std_17)
target_compile_features(cuinterval PUBLIC cuda_std_17)

source_group(
    TREE "${INCLUDE_PATH}"
    PREFIX "Header Files"
    FILES ${headers})

# set_target_properties(cuinterval PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

