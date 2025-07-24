set(EXAMPLES_PATH "cuinterval/examples")

set(headers
    "${EXAMPLES_PATH}/bisection.cuh"
)

# Add example procedures using the cuinterval library which are then used in testing code.
add_library(example_interval_procedures INTERFACE ${headers})
target_compile_features(example_interval_procedures INTERFACE cxx_std_17 cuda_std_17)
target_include_directories(example_interval_procedures INTERFACE "${cuinterval_SOURCE_DIR}/examples")
target_link_libraries(example_interval_procedures INTERFACE cuinterval)

# Add an example with the given name and optional cuda flags.
# Usage:
#        add_example(name)
#        add_example(name cuda_flag1 cuda_flag2 ...)
function(add_example name)
    add_executable(${name} ${name}.cu)
    target_compile_features(${name} PRIVATE cxx_std_20 cuda_std_20)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES native)
    target_link_libraries(${name} PRIVATE ${PROJECT_NAME})
    foreach(cuda_flag ${ARGN})
        target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_flag}>")
    endforeach()
endfunction()

add_example(bisection)
target_sources(bisection PRIVATE ${headers})

add_subdirectory(basic)
add_subdirectory(formatting)
