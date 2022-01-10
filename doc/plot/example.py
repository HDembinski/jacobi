from matplotlib import pyplot as plt
import numpy as np
from jacobi import jacobi


# function of one variable with auxiliary argument; returns a vector
def f(p, x):
    y = p + x
    return np.sin(y) / y


x = np.linspace(-10, 10, 1000)
fx = f(0, x)
fdx, fdex = jacobi(f, 0, x)

plt.plot(x, fx, label="f(x) = sin(x) / x")
plt.plot(x, fdx, ls="--", label="f'(x)")
plt.legend()

plt.savefig("doc/_static/example.svg")
