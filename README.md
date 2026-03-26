# typingkit

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python strong typing suite, along with Typed NumPy: Static shape typing and runtime shape validation.

> [!WARNING]
> Experimental & WIP.
> See [USAGE.md](USAGE.md) for more details.

## Installation

<details>

<summary>Install uv (optional, recommended)</summary>

Install [`uv`](https://docs.astral.sh/uv/), if not already.
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
If you don't want to use `uv`, skip to the next step.

**TL;DR: Just run**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

<details>

<summary>Install the package</summary>

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.
At present, the only required dependency is `numpy`.

Install the package from the PyPI release:

```shell
# Using uv
uv add typingkit

# Or with pip
pip3 install typingkit
```

To install from the latest commit:

```shell
uv add git+https://github.com/AshrithSagar/typingkit.git@main
```

</details>

## Usage

```python
from typing import Literal, TypeAlias, TypeVar

import numpy as np
from typingkit.numpy import TypedNDArray as NDArray

# Shape variables are just regular TypeVars
N = TypeVar("N", bound=int, default=int)
M = TypeVar("M", bound=int, default=int)

# Create aliases such as these, or use TypedNDArray directly
Vector: TypeAlias = NDArray[tuple[N], np.dtype[Any]]
Matrix: TypeAlias = NDArray[tuple[M, N], np.dtype[Any]]

v1 = Vector([1, 2, 3])  # Passes
v2 = Vector([4, 5, 6, 7])  # Also passes

v3 = NDArray[tuple[int]]([[8, 9]])
# Fails, since expected 1D array but passed in a 2D array

# Literal types also work
v4 = Vector[Literal[3]]([1, 2, 3])  # Passes
v5 = Vector[Literal[3]]([4, 5, 6, 7])  # Fails

v6 = Vector[Literal[3, 4]]([4, 5, 6, 7])  # Passes
# Allowed shapes are either (3,) or (4,)
```

See [USAGE.md](USAGE.md) for more details.

## License

This project falls under the [MIT License](LICENSE).
