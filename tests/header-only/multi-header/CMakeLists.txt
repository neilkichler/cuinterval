cmake_minimum_required(VERSION 3.25.2)

project(cuinterval-multi-header-test LANGUAGES CUDA)

# test multi-header with the basic example from the examples folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/basic/basic.cu DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# copy cuinterval folder and use it standalone
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/cuinterval DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(basic basic.cu)
target_include_directories(basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(basic PRIVATE cuda_std_20)
target_compile_options(basic PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(basic PROPERTIES CUDA_ARCHITECTURES native)
