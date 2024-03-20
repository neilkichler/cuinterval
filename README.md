<h1 align='center'>CuInterval</h1>

CuInterval is a CUDA [interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) library. It includes all fundamental and set-based interval operations of the [IEEE Standard for Interval Arithmetic](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7140721).
Other flavors, including decorations are not supported. All operations are correctly-rounded, given the limitations of the precision of the underlying [CUDA operation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#id200). The tightest interval is always a subset
of the computed interval. The lower and upper bounds of the basic operations are at most 3 [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place) away from the lower and upper bounds of the correct interval, respectively.

## Installation
> Please make sure that you have installed everything mentioned in the section [Build Requirements](#build-requirements).
```bash
git clone https://github.com/neilkichler/cuinterval.git
cd cuinterval
cmake --workflow --preset install
```

## Example
Have a look at the [examples folder](https://github.com/neilkichler/cuinterval/tree/main/examples).

## Documentation
The documentation is available [here](https://neilkichler.github.io/cuinterval).

## Build

### Build Requirements
We require C++17, CMake v3.21+, Ninja, and recent C++ and CUDA compilers.

```bash
apt install cmake ninja-build gcc
```

### Build and run tests
#### Using Workflows
```bash
cmake --workflow --preset dev
```
#### Using Presets
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```
#### Using regular CMake
```bash
cmake -S . -B build -GNinja
cmake --build build
./build/tests/tests
```
