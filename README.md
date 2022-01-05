# Jacobi

Fast numerical derivatives for real analytic functions with arbitrary round-off error.

## Features

- Robustly compute the generalised Jacobi matrix for an arbitrary real analytic mapping of $\mathcal{R}^k \to \mathcal{R}^{n_1} \times \dots \times \mathcal{R}^{n_\ell}$
- Derivative is computed to specified accuracy or until precision of function is reached
- Robust algorithm based on John D'Errico's DERIVEST: flawlessly works with functions that have large round-off error (internal computation in float32 precision or lower)
- Up to 100x faster than [numdifftools](https://pypi.org/project/numdifftools/) at equal precision
- Computes error estimates for derivatives
- Supports functions with arbitrary auxiliary arguments

## Example

```py
from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi


# function of one variable with auxiliary argument; returns a vector
def f(p, x):
    y = p + x
    return np.sin(y) / y


x = np.linspace(-10, 10, 1000)
fx = f(0, x)
fdx, fdex = jacobi(f, 0, x) # returns derivative and error estimate

plt.plot(x, fx, label="f(x) = sin(x) / x")
plt.plot(x, fdx, ls="--", label="f'(x)")
plt.legend()
plt.show()
```
![](doc/_static/example.svg)

## Comparison to numdifftools

#### Speed

![](doc/_static/speed.svg)

#### Precision

![](doc/_static/precision.svg)
