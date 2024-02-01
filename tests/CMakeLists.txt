include(FetchContent)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.0.1
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests tests.cu)

# I'm using C++17 in the test
target_compile_features(tests PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(tests PRIVATE cuinterval Boost::ut)

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.

# Register tests to ctest
add_test(NAME testcuinterval COMMAND tests) # Command can be a target


