set(CUINTERVAL_URL https://raw.githubusercontent.com/neilkichler/cuinterval/refs/heads/single-header/cuinterval.cuh)
file(DOWNLOAD ${CUINTERVAL_URL} ${CMAKE_CURRENT_SOURCE_DIR}/cuinterval/cuinterval.h)

add_library(cuinterval-single-header INTERFACE IMPORTED)
target_include_directories(cuinterval-single-header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# test single-header with the basic example from the examples folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../../examples/basic/basic.cu DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(system-header basic.cu)
target_link_libraries(system-header PRIVATE cuinterval-single-header)
target_compile_features(system-header PRIVATE cuda_std_20)
target_compile_options(system-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(system-header PROPERTIES CUDA_ARCHITECTURES native)
