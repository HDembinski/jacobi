from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi
from numdifftools import Derivative


# function of one variable with auxiliary argument; returns a vector
def f(x):
    return np.sin(x) / x


def fp(x):
    return np.cos(x) / x - np.sin(x) / x**2


x = np.linspace(-10, 10, 200)
fpx = fp(x)
fpx1, fpxe1 = jacobi(f, x, diagonal=True)
fpx2 = Derivative(lambda p: f(x + p))(0)

plt.figure(constrained_layout=True)
plt.plot(x, np.abs(fpx1 - fpx), ls="-", label="Jacobi")
plt.plot(x, np.abs(fpx2 - fpx), ls="--", label="numdifftools")
plt.fill_between(x, 0, fpxe1, alpha=0.5, facecolor="C0", label="error estimate")
plt.title("$f(x) = sin(x)/x,\\quad f'(x) = cos(x) / x - sin(x) / x^2$")
plt.legend(ncol=3, loc="upper center", frameon=False)
plt.semilogy()
plt.ylim(1e-17, 1e-13)
plt.ylabel("$|f'_\\mathrm{num}(x) - f'(x)|$")
plt.axhline(np.finfo(float).resolution, color="k", ls="--")

plt.savefig("doc/_static/precision.svg")
