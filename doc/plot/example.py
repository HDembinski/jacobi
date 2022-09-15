from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi


# function of one variable with auxiliary argument; returns a vector
def f(x):
    return np.sin(x) / x


x = np.linspace(-10, 10, 200)
fx = f(x)
fdx, fdxe = jacobi(f, x, diagonal=True)

plt.plot(x, fx, label="$f(x) = sin(x) / x$")
plt.plot(x, fdx, ls="--", label="$f'(x)$")
scale = 14
plt.fill_between(
    x,
    fdx - fdxe * 10**scale,
    fdx + fdxe * 10**scale,
    label=f"$f'(x)$ error estimate$\\times \\, 10^{{{scale}}}$",
    facecolor="C1",
    alpha=0.5,
)
plt.legend()

plt.savefig("doc/_static/example.svg")
