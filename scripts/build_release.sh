#!/usr/bin/env bash
set -euo pipefail

version="${1:-${NEW_VERSION:-}}"
if [[ -z "$version" ]]; then
  echo "usage: build_release.sh <version> (or set NEW_VERSION)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

staging=build/release
rm -rf "$staging" build/python-linux build/python-win
mkdir -p "$staging"/earth/python "$staging"/earth/otls build/python-linux build/python-win dist

cp src/*.hda "$staging"/earth/otls/
cp src/*.json "$staging"/
cp -r earth "$staging"/earth/python/earth
uv pip install --target "$staging"/earth/python --python 3.11 --no-deps trimesh
uv pip install --target build/python-linux --python 3.11 --no-deps dracopy
uv pip install --target build/python-win \
  --python-platform x86_64-pc-windows-msvc --python-version 3.11 \
  --only-binary :all: --no-deps dracopy
cp build/python-linux/DracoPy*.so "$staging"/earth/python/
cp build/python-win/DracoPy*.pyd build/python-win/msvcp140-*.dll "$staging"/earth/python/

rm -rf build/python-linux build/python-win
rm -rf "$staging"/earth/python/bin
rm -f "$staging"/earth/python/.lock
find "$staging" -name __pycache__ -type d -prune -exec rm -rf {} +

rm -rf dist
mkdir -p dist
(cd "$staging" && zip -r "../../dist/earth-v${version}.zip" .)

echo "Built dist/earth-v${version}.zip"
