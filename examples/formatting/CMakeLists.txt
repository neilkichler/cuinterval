add_executable(formatting formatting.cu)
target_compile_features(formatting PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(formatting PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(formatting PRIVATE ${PROJECT_NAME})
