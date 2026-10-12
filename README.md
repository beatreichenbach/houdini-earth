# Houdini Earth

Generate real-world meshes in SideFX Houdini using Google's 3D Tiles.

Earth is a set of nodes that build real-world geometry in Houdini from Google's
photorealistic 3D Tiles. It is inspired by the original
[EarthMeshHoudini](https://github.com/xjorma/EarthMeshHoudini) plugin.

## Nodes

| Node | Description |
|------|-------------|
| **Earth Mesh** | Downloads the 3D Tiles around a latitude/longitude and assembles them into a single mesh, with level of detail falling off by distance. |

The node builds the mesh at the origin in a local tangent space (`1` unit is
`1` meter, `X` points north and `Z` points east). It does not assign a material;
instead it writes the source texture path of each chunk to a `texture` primitive
attribute so you can bind it in your own shader network.

## Prerequisites

### Google Cloud Account

The node requires a Google Cloud API key with the **Map Tiles API** enabled. See
the [Google Cloud documentation](https://cloud.google.com/gcp) for how to create
a project and enable the API.

### Additional Python Packages

The library uses `pygltflib` to read GLTF files and `DracoPy` to decompress the
mesh compression used by Google's 3D Tiles:

```sh
hython -m pip install pygltflib DracoPy
```

## Installation

Earth is installed as a Houdini Package.

1. Go to the [Releases](https://github.com/beatreichenbach/houdini-earth/releases)
   page and download the latest `earth.zip`.
2. Navigate to your Houdini user preferences folder: `$HOME/houdini‹X›.‹Y›`
3. If you don't have a folder named `packages` in that directory, create it.
4. Extract the contents of the `.zip` file into the `packages` folder.
5. Restart Houdini. You can verify the installation by opening the **Package
   Browser** (**Windows > Package Browser**) and ensuring "Earth" is listed.

**Example Folder Structure:**

```text
├── houdini21.0
    ├── packages/
        ├── earth.json
        └── /earth
            ├── (HDA files)
            └── /python
                └── /earth
                    └── (Python library)
```

## Usage

Place the Earth Mesh node in your scene and set the **Google API Key**, a
**Cache Path** and the **Latitude** / **Longitude** of the area you want to
capture.

- **Min Error** / **Max Error** set the finest and coarsest level of detail.
- **Min Dist** / **Max Dist** control where that detail falls off. Everything
  within **Min Dist** is built at **Min Error**, everything at **Max Dist** at
  **Max Error**, interpolated in between. Nothing beyond **Max Dist** is
  downloaded.
- **Max Meshes** limits the number of downloaded tiles.
- **Clear Cache** removes the downloaded tiles from the cache path.

## About this Repository

| Directory | Description |
|-----------|-------------|
| `src/`    | Digital Assets and the package definition. During the release action they get packaged in the `earth.zip` archive. |
| `earth/`  | The Python library (tile download, GLB decoding, geometry assembly). |
| `node/`   | The Python and VEX snippets embedded in the HDAs. |
| `work/`   | Houdini files used to create icons, promotional materials and demos. |
| `build/`  | Build directory used to generate the release package. |

## Development

```sh
uv venv --python 3.11
uv sync
pre-commit install
```

For Houdini to import the library, the repository root (the directory
containing `earth/`) must be on `PYTHONPATH`.

## License

Copyright (c) 2026 Beat Reichenbach.
This project is licensed under the [GPLv3 License](LICENSE).
