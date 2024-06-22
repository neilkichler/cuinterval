include(FetchContent)

set(BOOST_UT_DISABLE_MODULE ON)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.0.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests 
    tests.cpp
    tests_ops.cu
    tests_ops.cuh
    tests_bisect.h 
    tests_bisect.cu 
    tests_bisect.cpp 
    tests_loop.h
    tests_loop.cu
    tests_loop.cpp
    generated/tests_generated.cpp
    generated/tests_c_xsc.cpp
    generated/tests_libieeep1788_elem.cpp
    generated/tests_libieeep1788_rec_bool.cpp
    generated/tests_libieeep1788_cancel.cpp
    generated/tests_libieeep1788_bool.cpp
    generated/tests_libieeep1788_set.cpp
    generated/tests_libieeep1788_num.cpp
    generated/tests_atan2.cpp
    generated/tests_mpfi.cpp
    generated/tests_intervalarithmeticjl.cpp
    generated/tests_filib.cpp
)

# Use C++20 in the tests
target_compile_features(tests PRIVATE cxx_std_20)
target_compile_features(tests PRIVATE cuda_std_20)

# Specify the CUDA architecture to compile to
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES native)

# Enable separate compilation
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with main library and testing library
target_link_libraries(tests PRIVATE cuinterval example_interval_procedures ut)

# Find and link with CUDA headers
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tests PRIVATE CUDA::cuda_driver)
target_link_libraries(tests PRIVATE CUDA::cudart)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PUBLIC "$<IF:$<CXX_COMPILER_ID:MSVC>,-openmp:llvm,OpenMP::OpenMP_CXX>")
endif()

target_compile_options(tests PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Register tests for use in ctest
add_test(NAME testcuinterval COMMAND tests)

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
if(ENABLE_TEST_COVERAGE)
    target_compile_options(tests PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(tests PUBLIC -fprofile-arcs -ftest-coverage)
endif()
