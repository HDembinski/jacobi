from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi


# function of one variable with auxiliary argument; returns a vector
def f(x):
    return np.sin(x) / x


x = np.linspace(-10, 10, 200)
fx = f(x)

# f(x) is a simple vectorized function, jacobian is diagonal
fdx, fdxe = jacobi(f, x, diagonal=True)
# fdxe is uncertainty estimate for derivative

plt.plot(x, fx, color="k", label="$f(x) = sin(x) / x$")
plt.plot(x, fdx, label="$f'(x)$ computed with jacobi")
scale = 14
plt.fill_between(
    x,
    fdx - fdxe * 10**scale,
    fdx + fdxe * 10**scale,
    label=f"$f'(x)$ error estimate$\\times \\, 10^{{{scale}}}$",
    facecolor="C0",
    alpha=0.5,
)
plt.legend()

plt.savefig("doc/_static/example.svg")
