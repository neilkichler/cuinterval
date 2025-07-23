
set(EXAMPLES_PATH "cuinterval/examples")

set(headers
    "${EXAMPLES_PATH}/bisection.cuh"
)

# Add example program with user interaction to test the library interactively.
add_executable(examples bisection.cu ${headers})
target_compile_features(examples PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(examples PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(examples PRIVATE cuinterval)

# Add example procedures using the cuinterval library which are then used in testing code.
add_library(example_interval_procedures INTERFACE ${headers})
target_compile_features(example_interval_procedures INTERFACE cxx_std_17 cuda_std_17)
target_include_directories(example_interval_procedures INTERFACE "${cuinterval_SOURCE_DIR}/examples")
target_link_libraries(example_interval_procedures INTERFACE cuinterval)

add_subdirectory(basic)
add_subdirectory(formatting)
