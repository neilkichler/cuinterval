<h1 align='center'>CuInterval</h1>

CuInterval is a CUDA [interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) library. It includes all fundamental and set-based interval operations of the [IEEE Standard for Interval Arithmetic](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7140721).
Other flavors, including decorations are not supported. 
## Supported Operations

The following operations are implemented as CUDA kernels. All operations are correctly-rounded, given the limitations of the precision of the underlying [CUDA operation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#id200). The tightest interval is always a subset
of the computed interval. The lower and upper bounds of the basic operations are at most 3 [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place) away from the lower and upper bounds of the correct interval, respectively.
The error for a paricular operation is given below.

<details>
<summary>Basic Operations</summary>

| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| pos                | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| neg                | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| add                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| sub                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| mul                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| div                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| recip              | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| sqr                | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| sqrt               | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| fma                | $f: \mathbb{IR} \times \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |

</details>

<details>
<summary>Power functions</summary>

| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| cbrt      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 1            |
| exp       | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| exp2      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| exp10     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| expm1     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| log       | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| log2      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| log10     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| log1p     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| pown      | $f: \mathbb{IR} \times \mathbb{N} \rightarrow \mathbb{IR}$ | 1 |
| pow       | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 1 |
| rootn     | $f: \mathbb{IR} \times \mathbb{N} \rightarrow \mathbb{IR}$ | 2 |

</details>

<details>
<summary>Trigonometric functions</summary>
  
| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| sin       | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| cos       | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| tan       | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| asin      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| acos      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atan      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atan2     | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| sinpi     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| cospi     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |

</details>

<details>
<summary>Hyperbolic functions</summary>
  
| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| sinh      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| cosh      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| tanh      | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| asinh     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| acosh     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atanh     | $f: \mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |

</details>


<details>
<summary>Integer functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| floor              | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| ceil               | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| trunc              | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| sign               | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| roundTiesToEven    | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| roundTiesToAway    | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |

</details>

<details>
  <summary>Absmax functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| abs                | $f: \mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| min                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| max                | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |

</details>

<details>
<summary>Numeric functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| inf                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| sup                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| mid                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| wid                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| rad                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| mag                | $f: \mathbb{IR} \rightarrow T$              | 0            |
| mig                | $f: \mathbb{IR} \rightarrow T$              | 0            |

</details>

<details>
<summary>Boolean functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| equal              | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| subset             | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| interior           | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| disjoint           | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| isEmpty            | $f: \mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| isEntire           | $f: \mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| less               | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| strictLess         | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| precedes           | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| strictPrecedes     | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| isMember           | $f: T \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| isSingleton        | $f: \mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| isCommonInterval   | $f: \mathbb{IR} \rightarrow \mathbb{B}$     | 0            |

</details>


<details>
<summary>Set operations</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| intersection       | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| convexHull         | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
</details>


<details>
<summary>Cancellative add and subtract</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| cancelMinus        | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |
| cancelPlus         | $f: \mathbb{IR} \times \mathbb{IR} \rightarrow \mathbb{IR}$ | 0            |

</details>

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
