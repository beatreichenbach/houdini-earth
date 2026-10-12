# Contributing Guide

## Development

To get started:

```sh
uv venv --python 3.11
uv sync
pre-commit install
```

Run the checks:

```sh
ruff format .
ruff check --fix .
ty check
pytest
```

### Releasing Changes

Releases are built by pushing a `v*` tag, which triggers the release workflow.
Bump `version` in `pyproject.toml` and `__version__` in `earth/__init__.py` before tagging.

## Working with Houdini

Build the runtime dependencies and symlink the sources so Houdini always loads
the local files:

```sh
uv pip install --target build/python dracopy pygltflib
mkdir -p build/otls build/python
ln -sfn "$PWD"/src/*.hda build/otls/
ln -sfn "$PWD"/earth build/python/earth
```

Then point a Houdini package at `build/otls` via `HOUDINI_OTLSCAN_PATH` and at
`build/python` via `PYTHONPATH`.
