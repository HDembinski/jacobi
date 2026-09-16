See the [Scikit-HEP Developer introduction][skhep-dev-intro] for a detailed
description of best practices for developing Scikit-HEP packages.

[skhep-dev-intro]: https://scikit-hep.org/developer/intro

# Contributing

## Setting up a development environment

### Nox

The fastest way to start with development is to use nox. If you don't have nox,
you can use `uvx nox` to run it without installing, or `uv tool install nox`. If
you don't have uv (or pipx, which is similar), then you can install it with
`pip install uv` (the only case where installing an application with regular pip
is reasonable). If you use macOS, then uv, prek, and nox are all in brew, use
`brew install uv prek nox`.

To use, run `nox`. This will lint and test using the Python version nox runs on.
You can also run specific jobs:

```console
$ nox -l # List all the defined sessions
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s tests -P 3.12  # Python tests on a specific Python version
$ nox -s minimums  # Python tests with the lowest supported dependency versions
$ nox -s cov  # Python tests with an HTML coverage report
$ nox -s bench  # Benchmarks
$ nox -s docs  # Build and serve the docs
$ nox -s plots  # Regenerate the figures used in the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up a temporary virtual
environment for each run.

### uv

For extended development, you can set up a development environment with uv:

```bash
uv sync --group test --group docs
uv run pytest
```

## Post setup

You should prepare prek, which will help you by checking that commits pass
required checks:

```bash
uv tool install prek # or brew install prek on macOS
prek install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `prek` (changes only) or `prek -a` to check even
without installing the hook.

## Testing

Use pytest to run the unit checks:

```bash
uv run pytest
```
