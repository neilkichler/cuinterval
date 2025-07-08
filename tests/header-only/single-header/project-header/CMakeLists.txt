set(CUINTERVAL_URL https://raw.githubusercontent.com/neilkichler/cuinterval/refs/heads/single-header/cuinterval.cuh)
file(DOWNLOAD ${CUINTERVAL_URL} ${CMAKE_CURRENT_SOURCE_DIR}/cuinterval.cuh)

# test single-header with the basic example from the examples folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../../examples/basic/basic.cu DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# Adjust basic.u to use single-header cuinterval.cuh as a project header
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/basic.cu f)
string(REPLACE "#include <cuinterval/cuinterval.h>" "#include \"cuinterval.cuh\"" f "${f}")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/basic.cu "${f}")

add_executable(project-header basic.cu)
target_compile_features(project-header PRIVATE cuda_std_20)
target_compile_options(project-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(project-header PROPERTIES CUDA_ARCHITECTURES native)
