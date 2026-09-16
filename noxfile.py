#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

"""Nox runner."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PROJECT = nox.project.load_toml()
MINIMUM_PYTHON = PROJECT["project"]["requires-python"].removeprefix(">=")

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("prek")
    session.run(
        "prek", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """Run Pylint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install("-e.", "pylint>=3.2")
    session.run("pylint", "jacobi", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps)
    session.run("pytest", *session.posargs)


@nox.session(python=MINIMUM_PYTHON, venv_backend="uv", default=False)
def minimums(session: nox.Session) -> None:
    """Run the tests with the minimum versions of the direct dependencies."""
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps, "--resolution=lowest-direct")
    session.run("uv", "pip", "list")
    session.run("pytest", *session.posargs)


@nox.session(default=False)
def cov(session: nox.Session) -> None:
    """Run the tests with coverage and write an HTML report to "htmlcov"."""
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps)
    session.run(
        "pytest",
        "--cov",
        "--cov-report=term-missing",
        "--cov-report=html",
        *session.posargs,
    )


@nox.session(default=False)
def bench(session: nox.Session) -> None:
    """Run the benchmarks."""
    bench_deps = nox.project.dependency_groups(PROJECT, "bench")
    session.install("-e.", *bench_deps)
    session.run("pytest", "bench", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs.

    Serves the docs when run interactively; pass --non-interactive to nox to
    avoid serving. The first positional argument is the target directory.
    """
    doc_deps = nox.project.dependency_groups(PROJECT, "docs")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.", *doc_deps, *(["sphinx-autobuild"] if serve else []))

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-W",  # warnings are errors
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(reuse_venv=True, default=False)
def plots(session: nox.Session) -> None:
    """Regenerate the figures in docs/_static."""
    plot_deps = nox.project.dependency_groups(PROJECT, "plot")
    session.install("-e.", *plot_deps)
    for script in sorted((DIR / "docs" / "plot").glob("*.py")):
        session.run("python", str(script))


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
