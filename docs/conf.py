"""Sphinx configuration."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

DIR = Path(__file__).parent.resolve()

# index.rst is generated from index.rst.in and the README, so that the
# front page of the documentation and the README stay in sync.
readme = (DIR.parent / "README.rst").read_text(encoding="utf-8")
readme = readme.replace("https://hdembinski.github.io/jacobi/_images/", "_static/")
stub = (DIR / "index.rst.in").read_text(encoding="utf-8")
index = DIR / "index.rst"
content = stub + "\n" + readme
# Only write when needed, otherwise sphinx-autobuild rebuilds in a loop.
if not index.is_file() or index.read_text(encoding="utf-8") != content:
    index.write_text(content, encoding="utf-8")

project = "jacobi"
copyright = "2020, Hans Dembinski"
author = "Hans Dembinski"
version = release = importlib.metadata.version("jacobi")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

# Types are documented in the numpy-style docstrings.
autodoc_typehints = "none"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Render parameter and return types verbatim instead of cross-referencing them.
napoleon_use_param = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/hdembinski/jacobi",
    "source_branch": "main",
    "source_directory": "docs/",
}
