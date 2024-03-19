cmake_minimum_required(VERSION 3.21)

project(
    cuinterval
    VERSION 0.0.1
    DESCRIPTION "CUDA interval library"
    LANGUAGES CXX CUDA)

option(CUINTERVAL_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(CUINTERVAL_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
option(CUINTERVAL_ENABLE_SANITIZER_UNDEFINED "Enable undefined behavior sanitizer" OFF)
option(CUINTERVAL_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
option(CUINTERVAL_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)

include(cmake/sanitizers.cmake)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Better support of clang based tools
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Enable position independent code
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Include example programs
  add_subdirectory(examples)

  # Testing only available if this is the main project.
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
  include(CTest)

  add_subdirectory(tests)
endif()

# Include main library
add_subdirectory(src)

