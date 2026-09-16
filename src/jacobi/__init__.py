"""Fast numerical derivatives for analytic functions with arbitrary round-off error."""

from __future__ import annotations

from ._jacobi import jacobi
from ._propagate import propagate
from ._version import version as __version__

__all__ = ["__version__", "jacobi", "propagate"]
