#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build/earth/otls build/earth/python
cp src/earth.json build/

uv pip install --target build/earth/python --python 3.11 --no-deps trimesh dracopy

rm -rf build/earth/python/earth
ln -sfn "$PWD/earth" build/earth/python/earth
rm -f build/earth/otls/*.hda
ln -sfn "$PWD"/src/*.hda build/earth/otls/

rm -rf build/earth/python/bin
rm -f build/earth/python/.lock

echo "Dev package ready. Add '$PWD/build' to HOUDINI_PACKAGE_DIR."
