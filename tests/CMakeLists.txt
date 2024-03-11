include(FetchContent)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.0.1
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests tests.cpp tests_bisect.h tests_bisect.cu test_ops.cuh)

# Use C++20 in the tests
target_compile_features(tests PRIVATE cxx_std_20)
target_compile_features(tests PRIVATE cuda_std_20)

# Specify the CUDA architecture to compile to
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES "61;70;86")

# Link with main library and testing library
target_link_libraries(tests PRIVATE cuinterval example_interval_procedures Boost::ut)

find_package(OpenMP)
# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Found OpenMP CXX")
endif()

# Register tests for use in ctest
add_test(NAME testcuinterval COMMAND tests)

