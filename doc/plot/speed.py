from jacobi import jacobi
import numpy as np
from numdifftools import Derivative
from timeit import timeit
from matplotlib import pyplot as plt

fn = [
    "x ** 2",
    "exp(x)",
    "ones_like(x)",
    "mean(x ** 2)",
]


def ja(f, x):
    return jacobi(lambda p: f(p + x), 0)[0]


def nd(f, x):
    return Derivative(lambda p: f(p + x))(0)


fig, ax = plt.subplots(
    2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True
)
for i, fi in enumerate(fn):
    t = {}
    t["jacobi"] = []
    t["numdifftools"] = []
    nmax = 5
    n = (10 ** np.linspace(0, nmax, nmax + 1)).astype(int)
    f = eval("lambda x: " + fi, np.__dict__)
    for ni in n:
        print(i, ni)
        x = np.linspace(0.1, 10, ni)
        r = timeit(lambda: ja(f, x), number=500 // ni + 10)
        t["jacobi"].append(r)
        r = timeit(lambda: nd(f, x), number=500 // ni + 1)
        t["numdifftools"].append(r)
    plt.sca(ax.flat[i])
    for k, v in t.items():
        ls = "-" if k == "jacobi" else "--"
        plt.plot(n, v, ls=ls, label=k)
    r = np.divide(t["numdifftools"], t["jacobi"])
    plt.plot(n, r, ls=":", color="k", lw=3, label="time ratio")
    for ni, ri in zip(n, r):
        plt.text(
            ni, ri * 0.5, f"{ri:.0f}" if i != 3 else f"{ri:.1f}", va="top", ha="center"
        )
    plt.legend(title=fi, loc="lower left", frameon=False)
plt.loglog()
fig.supxlabel("n")
fig.supylabel("t/sec")
plt.savefig("doc/_static/speed.svg")
