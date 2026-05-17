# HyperGraph

**Reverse-mode second-order automatic differentiation for Python.**

HyperGraph computes exact **gradients** and **Hessians** of scalar expressions using a computational-graph approach based on reverse-mode automatic differentiation. The core is implemented in C++ with [Eigen](https://eigen.tuxfamily.org/) and exposed to Python via [pybind11](https://github.com/pybind/pybind11), making it both fast and easy to use.

[![CI](https://github.com/oberbichler/HyperGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/oberbichler/HyperGraph/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hypergraph)](https://pypi.org/project/hypergraph)
[![Python](https://img.shields.io/pypi/pyversions/hypergraph)](https://pypi.org/project/hypergraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> This is an experimental project. It implements the graph-based edge-pushing algorithm from Gower & Mello (2010) as a standalone library. The approach works well but is not as widely adopted or battle-tested as other techniques.

---

## Features

- **Second-order derivatives** — computes both the gradient vector and the full Hessian matrix in a single pass
- **High performance** — C++17 core with Eigen, optional BLAS acceleration
- **NumPy integration** — HyperGraph variables work seamlessly with `np.sin`, `np.cos`, `np.sqrt`, `np.dot`, `np.cross`, `np.linalg.norm`, and more
- **Comprehensive math library** — arithmetic, trigonometric, hyperbolic, inverse-hyperbolic, exponential, logarithmic, and power functions
- **Cross-platform** — tested on Linux, macOS, and Windows with Python 3.12 and 3.13

## Installation

Install from [PyPI](https://pypi.org/project/hypergraph):

```bash
pip install hypergraph
```

## Quick Start

```python
import hypergraph as hg
import numpy as np

# Create a computation graph
graph = hg.HyperGraph()

# Define variables
x, y = graph.new_variables([2.0, 3.0])

# Build an expression: f(x, y) = x² · sin(y)
f = x ** 2 * np.sin(y)

# Compute gradient and Hessian
graph.compute(f)

print("f =", f.value)           # 4.0 * sin(3.0) ≈ 0.5645
print("∇f =", graph.g())        # [2x·sin(y), x²·cos(y)]
print("Hf =\n", graph.h())      # Hessian matrix
```

## Usage

### Creating Variables

```python
graph = hg.HyperGraph()

# Single variable
a = graph.new_variable(5.0)

# Multiple variables at once
x, y, z = graph.new_variables([1.0, 2.0, 3.0])
```

### Building Expressions

Variables support all standard arithmetic operators and can be used directly with NumPy functions:

```python
# Arithmetic
f = (x + y) * z - 1.0

# NumPy functions
f = np.sin(x) * np.cos(y) + np.sqrt(z)

# Vector operations
a = np.array([x, y, z])
b = np.array([1.0 * x, 2.0 * y, 3.0 * z])
f = np.linalg.norm(np.cross(a, b))
```

### Computing Derivatives

```python
# Evaluate gradient and Hessian for a scalar expression
graph.compute(f)

# Gradient (as NumPy vector)
g = graph.g()                      # shape: (n,)

# Hessian (upper-triangular by default)
h = graph.h()                      # shape: (n, n), upper-triangular
h_full = graph.h(full=True)        # shape: (n, n), symmetric

# Write into pre-allocated arrays to avoid allocation
g_out = np.empty(n)
h_out = np.empty((n, n))
graph.g(out=g_out)
graph.h(out=h_out)
```

## Supported Operations

| Category | Operations |
|---|---|
| **Arithmetic** | `+`, `-`, `*`, `/`, `**`, `+=`, `-=`, `*=`, `/=`, unary `-` |
| **Comparison** | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| **Trigonometric** | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `atan2` |
| **Hyperbolic** | `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` |
| **Exponential** | `exp`, `log` |
| **Power / Root** | `pow` (`**`), `square`, `sqrt`, `abs` |

Trigonometric and root functions can be called via NumPy (`np.sin(x)`, `np.sqrt(x)`, …) or as methods on a variable (`x.sin()`, `x.sqrt()`, …). The `atan2` function is available as `hg.atan2(y, x)`.

## Building from Source

HyperGraph uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) and requires a C++17 compiler, CMake ≥ 3.24, and pybind11. [Eigen 3.4](https://eigen.tuxfamily.org/) is fetched automatically during the build.

```bash
# Clone the repository
git clone https://github.com/oberbichler/HyperGraph.git
cd HyperGraph

# Install with uv (recommended)
uv sync

# Or install with pip
pip install .

# Run tests
pytest
```

## Credits

- "Hessian Matrices via Automatic Differentiation", Gower and Mello 2010
- [HAD by Tzu-Mao Li](https://github.com/BachiLi/had)

## Citation

If you use HyperGraph in your research, please cite:

```bibtex
@misc{HyperGraph,
  author = {Thomas Oberbichler},
  title  = {HyperGraph},
  url    = {https://github.com/oberbichler/HyperGraph},
  year   = {2019}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
