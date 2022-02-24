Jacobi
======

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
- Up to 1200x faster than `numdifftools <https://pypi.org/project/numdifftools>`_ at equivalent precision
- Returns error estimates for derivatives
- Supports arbitrary auxiliary function arguments
- Lightweight package, only depends on numpy

Planned features
----------------

- Compute the Hessian matrix numerically with the same algorithm
- Further generalize the calculation to support function arguments with shape (N, K), in that case compute the Jacobi matrix for each of the K vectors of length N

Example
-------

.. code-block:: python

  from matplotlib import pyplot as plt
  import numpy as np
  from jacobi import jacobi


  # function of one variable with auxiliary argument; returns a vector
  def f(dx, x):
      y = x + dx
      return np.sin(y) / y


  x = np.linspace(-10, 10, 1000)
  fx = f(0, x)
  fdx, fdex = jacobi(f, 0, x)

  plt.plot(x, fx, label="f(x) = sin(x) / x")
  plt.plot(x, fdx, ls="--", label="f'(x)")
  plt.legend()

.. image:: https://hdembinski.github.io/jacobi/_images/example.svg

Comparison to numdifftools
--------------------------

Speed
^^^^^

Jacobi makes better use of vectorised computation than numdifftools. Smaller run-time is better (and ratio > 1).

.. image:: https://hdembinski.github.io/jacobi/_images/speed.svg

Precision
^^^^^^^^^

The machine precision is indicated by the dashed line.

.. image:: https://hdembinski.github.io/jacobi/_images/precision.svg
