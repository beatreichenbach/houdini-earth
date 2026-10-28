# Contributing Guide

## Development

To get started:

```sh
uv venv --python 3.11
uv sync --extra dev
uv run pre-commit install
```

Run the checks:

```sh
uv run ruff format .
uv run ruff check --fix .
uv run ty check
uv run pytest
```

## Built-in Packages

Houdini already ships some of the dependencies, so they are not bundled in the release and do not need to be installed:

- `requests` (tile downloads)
- `numpy` (geometry and tile math)
- `pillow` (texture handling in `trimesh`)

See the [Houdini third-party libraries](https://www.sidefx.com/docs/houdini/licenses/) page for the full list.

## Working with Houdini

Set up the development package in `build/`:

```sh
./scripts/build_dev.sh
```

Point Houdini at the package by adding `build/` to `HOUDINI_PACKAGE_DIR`:

## Releasing Changes

Releases are published with [python-semantic-release](https://python-semantic-release.readthedocs.io):

```sh
uv run semantic-release version && uv run semantic-release publish
```
