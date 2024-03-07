
set(EXAMPLES_PATH "cuinterval/examples")

set(headers
    "${EXAMPLES_PATH}/bisection.cuh"
)

# Add example program with user interaction to test the library interactively.
add_executable(examples bisection.cu ${headers})
target_compile_features(examples PRIVATE cxx_std_20)
target_compile_features(examples PRIVATE cuda_std_20)
target_link_libraries(examples PRIVATE cuinterval)

# Add example procedures using the cuinterval library which are then used in testing code.
add_library(example_interval_procedures ${headers})
target_compile_features(example_interval_procedures PUBLIC cxx_std_20)
target_compile_features(example_interval_procedures PUBLIC cuda_std_20)
target_include_directories(example_interval_procedures PUBLIC "${cuinterval_SOURCE_DIR}/examples")
target_link_libraries(example_interval_procedures PRIVATE cuinterval)

