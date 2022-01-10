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

- Robustly compute the generalised Jacobi matrix for an arbitrary real analytic mapping of ℝⁿ → ℝⁱ¹ × ... × ℝⁱⁿ
- Derivative is computed to specified accuracy or until precision of function is reached
- Algorithm based on John D'Errico's `DERIVEST <https://de.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation>`_: flawlessly works even with functions that have large round-off error
- Up to 1200x faster than `numdifftools <https://pypi.org/project/numdifftools>`_ at equivalent precision
- Returns error estimates for derivatives
- Supports calculation of derivative up to target precision (speed-up)
- Supports arbitrary auxiliary function arguments
- Lightweight package, only depends on numpy

Example
-------

.. code-block:: python

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

.. image:: https://hdembinski.github.io/jacobi/_images/example.svg

Comparison to numdifftools
--------------------------

Speed
^^^^^

Jacobi makes better use of vectorised computation than numdifftools.

.. image:: https://hdembinski.github.io/jacobi/_images/speed.svg

Precision
^^^^^^^^^

The machine precision is indicated by the dashed line.

.. image:: https://hdembinski.github.io/jacobi/_images/precision.svg
