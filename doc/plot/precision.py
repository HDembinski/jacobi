from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi
from numdifftools import Derivative


# function of one variable with auxiliary argument; returns a vector
def f(x):
    return np.sin(x) / x


def fp(x):
    return np.cos(x) / x - np.sin(x) / x**2


x = np.linspace(-10, 10, 1000)
fpx = fp(x)
fpx1, fpxe1 = jacobi(f, x, diagonal=True)
fpx2 = Derivative(lambda p: f(x + p))(0)

plt.figure(constrained_layout=True)
plt.plot(x, np.abs(fpx1 / fpx - 1), ls="-", label="Jacobi")
plt.plot(x, np.abs(fpx2 / fpx - 1), ls="--", label="numdifftools")
plt.title("relative deviation of numerical from true derivative")
plt.legend(title="f(x) = sin(x)/x", ncol=2)
plt.semilogy()
plt.ylim(1e-16, 1e-11)
plt.axhline(np.finfo(float).resolution, color="k", ls="--")

plt.savefig("doc/_static/precision.svg")
