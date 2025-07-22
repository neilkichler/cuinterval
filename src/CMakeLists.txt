set(INCLUDE_PATH "${cuinterval_SOURCE_DIR}/include")
set(LIB_PATH "${INCLUDE_PATH}/cuinterval")

set(headers 
    "${LIB_PATH}/arithmetic/intrinsic.cuh"
    "${LIB_PATH}/arithmetic/basic.cuh"
    "${LIB_PATH}/cuinterval.h"
    "${LIB_PATH}/format.h"
    "${LIB_PATH}/numbers.h"
)

add_library(cuinterval "main.cu" ${headers})

target_include_directories(
    cuinterval PUBLIC $<BUILD_INTERFACE:${INCLUDE_PATH}>
                      $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# C++20 is required for all users of the library
target_compile_features(cuinterval PUBLIC cxx_std_20 cuda_std_20)

target_compile_options(cuinterval PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")

source_group(
    TREE "${INCLUDE_PATH}"
    PREFIX "Header Files"
    FILES ${headers})

include(FetchContent)
FetchContent_Declare(
  PackageProject
  GIT_REPOSITORY https://github.com/TheLartians/PackageProject.cmake.git
  GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(PackageProject)

packageProject(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    NAMESPACE ${PROJECT_NAME}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    INCLUDE_DIR ${INCLUDE_PATH}
    INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
