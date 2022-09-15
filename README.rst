.. |jacobi| image:: https://hdembinski.github.io/jacobi/_images/logo.svg
   :alt: jacobi

|jacobi|
========

.. image:: https://img.shields.io/pypi/v/jacobi
  :target: https://pypi.org/project/jacobi
.. image:: https://img.shields.io/badge/github-docs-success
  :target: https://hdembinski.github.io/jacobi
.. image:: https://img.shields.io/badge/github-source-blue
  :target: https://github.com/HDembinski/jacobi

Fast numerical derivatives for real analytic functions with arbitrary round-off error.

Features
--------

- Robustly compute the generalised Jacobi matrix for an arbitrary real analytic mapping ℝⁿ → ℝⁱ¹ × ... × ℝⁱⁿ
- Derivative is either computed to specified accuracy (to save computing time) or until maximum precision of function is reached
- Algorithm based on John D'Errico's `DERIVEST <https://de.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation>`_: works even with functions that have large round-off error
- Up to 1000x faster than `numdifftools <https://pypi.org/project/numdifftools>`_ at equivalent precision
- Returns error estimates for derivatives
- Supports arbitrary auxiliary function arguments
- Perform statistical error propagation based on numerically computed jacobian
- Lightweight package, only depends on numpy

Planned features
----------------

- Compute the Hessian matrix numerically with the same algorithm
- Further generalize the calculation to support function arguments with shape (N, K), in that case compute the Jacobi matrix for each of the K vectors of length N

Examples
--------

.. code-block:: python

  from matplotlib import pyplot as plt
  import numpy as np
  from jacobi import jacobi


  def f(x):
      return np.sin(x) / x

  x = np.linspace(-10, 10, 1000)

  fx = f(x)

  # f(x) is a simple vectorized function, jacobian is diagonal
  fdx, fdxe = jacobi(f, x, diagonal=True)
  # fdxe is uncertainty estimate for derivative

  plt.plot(x, fx, label="f(x) = sin(x) / x")
  plt.plot(x, fdx, ls="--", label="f'(x)")
  plt.legend()

.. image:: https://hdembinski.github.io/jacobi/_images/example.svg

.. code-block:: python

  from jacobi import propagate
  import numpy as np
  from scipy.special import gamma


  # arbitrarily complex function that calls compiled libraries, numba-jitted code, etc.
  def fn(x):
      r = np.empty(3)
      r[0] = 1.5 * np.exp(-x[0] ** 2)
      r[1] = gamma(x[1] ** 3.1)
      r[2] = np.polyval([1, 2, 3], x[0])
      return r  # x and r have different lengths

  # fn accepts a parameter vector x, which has an associated covariance matrix xcov
  x = [1.0, 2.0]
  xcov = [[1.1, 0.1], [0.1, 2.3]]
  y, ycov = propagate(fn, x, xcov)  # y=f(x) and ycov = J xcov J^T


Comparison to numdifftools
--------------------------

Speed
^^^^^

Jacobi makes better use of vectorized computation than numdifftools and converges rapidly if the derivative is trivial. This leads to a dramatic speedup in some cases.

Smaller run-time is better (and ratio > 1).

.. image:: https://hdembinski.github.io/jacobi/_images/speed.svg

Precision
^^^^^^^^^

The machine precision is indicated by the dashed line.

.. image:: https://hdembinski.github.io/jacobi/_images/precision.svg
