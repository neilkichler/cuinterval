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
| pos                | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| neg                | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| add                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| sub                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| mul                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| div                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| recip              | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| sqr                | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| sqrt               | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 0            |
| fma                | $\mathbb{IR}^3 \rightarrow \mathbb{IR}$ | 0            |

</details>

<details>
<summary>Power functions</summary>

| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| pown      | $\mathbb{IR} \times \mathbb{N} \rightarrow \mathbb{IR}_{\ge \mathbf{0}}$ | 1 |
| pow       | $\mathbb{IR}^2 \rightarrow \mathbb{IR}_{\ge \mathbf{0}}$ | 1 |
| rootn     | $\mathbb{IR}_{\ge \mathbf{0}} \times \mathbb{N} \rightarrow \mathbb{IR}$ | 2 |
| cbrt      | $\mathbb{IR}_{\ge \mathbf{0}} \rightarrow \mathbb{IR}$     | 1            |
| exp       | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| exp2      | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| exp10     | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| expm1     | $\mathbb{IR} \rightarrow \mathbb{IR}$     | 3            |
| log       | $\mathbb{IR}_{\ge \mathbf{0}} \rightarrow \mathbb{IR}$     | 3            |
| log2      | $\mathbb{IR}_{\ge \mathbf{0}} \rightarrow \mathbb{IR}$     | 3            |
| log10     | $\mathbb{IR}_{\ge \mathbf{0}} \rightarrow \mathbb{IR}$ | 3 |
| log1p     | $\mathbb{IR}_{\ge \mathbf{-1}} \rightarrow \mathbb{IR}$ | 3 |

</details>

<details>
<summary>Trigonometric functions</summary>
  
| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| sin       | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| cos       | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| tan       | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| asin      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| acos      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atan      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atan2     | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 3 |
| sinpi     | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| cospi     | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |

</details>

<details>
<summary>Hyperbolic functions</summary>
  
| Operation | Function Description                        | Error [ulps] |
|-----------|---------------------------------------------|--------------|
| sinh      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| cosh      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| tanh      | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 2 |
| asinh     | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| acosh     | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |
| atanh     | $\mathbb{IR} \rightarrow \mathbb{IR}$ | 3 |

</details>


<details>
<summary>Integer functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| sign               | $\mathbb{IR} \rightarrow \\{-1, 0, 1\\}$      | 0            |
| ceil               | $\mathbb{IR} \rightarrow \mathbb{Z}$     | 0            |
| floor              | $\mathbb{IR} \rightarrow \mathbb{Z}$     | 0            |
| trunc              | $\mathbb{IR} \rightarrow \mathbb{Z}$     | 0            |
| roundTiesToEven    | $\mathbb{IR} \rightarrow \mathbb{Z}$     | 0            |
| roundTiesToAway    | $\mathbb{IR} \rightarrow \mathbb{Z}$     | 0            |

</details>

<details>
  <summary>Absmax functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| abs                | $\mathbb{IR} \rightarrow \mathbb{IR}_{\ge \mathbf{0}}$     | 0            |
| min                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| max                | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |

</details>

<details>
<summary>Numeric functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| inf                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| sup                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| mid                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| wid                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| rad                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| mag                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |
| mig                | $\mathbb{IR} \rightarrow \mathbb{R}$        | 0            |

</details>

<details>
<summary>Boolean functions</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| equal              | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| subset             | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| interior           | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| disjoint           | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| isEmpty            | $\mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| isEntire           | $\mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| less               | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| strictLess         | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| precedes           | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| strictPrecedes     | $\mathbb{IR}^2 \rightarrow \mathbb{B}$ | 0            |
| isMember           | $\mathbb{R} \times \mathbb{IR} \rightarrow \mathbb{B}$ | 0            |
| isSingleton        | $\mathbb{IR} \rightarrow \mathbb{B}$     | 0            |
| isCommonInterval   | $\mathbb{IR} \rightarrow \mathbb{B}$     | 0            |

</details>


<details>
<summary>Set operations</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| intersection       | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| convexHull         | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
</details>


<details>
<summary>Cancellative add and subtract</summary>
  
| Operation          | Function Description                        | Error [ulps] |
|--------------------|---------------------------------------------|--------------|
| cancelMinus        | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |
| cancelPlus         | $\mathbb{IR}^2 \rightarrow \mathbb{IR}$ | 0            |

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
