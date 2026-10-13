# Contributing Guide

## Development

To get started:

```sh
uv venv --python 3.11
uv sync --extra dev
pre-commit install
```

Run the checks:

```sh
ruff format .
ruff check --fix .
ty check
pytest
```

## Built-in Packages

Houdini already ships some of the dependencies, so they are not bundled in the release and do not need to be installed:

- `requests` — tile downloads
- `numpy` — geometry and tile math
- `pillow` — texture handling in `trimesh`

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
semantic-release version && semantic-release publish
```
