# Houdini Earth

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/beatreichenbach/houdini-earth)
[![License](https://img.shields.io/github/license/beatreichenbach/houdini-earth.svg)](https://github.com/beatreichenbach/houdini-earth/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type%20checked-ty-261230.svg)](https://github.com/astral-sh/ty)

Earth builds real-world geometry in SideFX Houdini from Google's photorealistic 3D Tiles.
It is inspired by the original [EarthMeshHoudini](https://github.com/xjorma/EarthMeshHoudini) plugin.

![Render](.github/assets/render.jpg)

## Installation

Earth is installed as a Houdini Package.

1. Go to the [Releases](https://github.com/beatreichenbach/houdini-earth/releases) page and download the latest `earth.zip`.
2. Navigate to your Houdini user preferences folder: `$HOME/houdini‹X›.‹Y›`
3. If you don't have a folder named `packages` in that directory, create it.
4. Extract the contents of the `.zip` file into the `packages` folder.
5. Restart Houdini. Verify the installation in the [Package Browser](https://www.sidefx.com/docs/houdini/ref/windows/package_browser.html).

**Example Folder Structure:**

```text
├── houdini21.0
    ├── packages/
        ├── earth.json
        └── /earth
```

### Google Cloud Account

The node requires a Google Cloud API key with the **Map Tiles API** and **Maps Elevation API** enabled.
See the [Google Cloud documentation](https://cloud.google.com/gcp) for how to create a project and enable the API.

Test if your key supports the Map Tiles API by trying this link:
<https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_API_KEY>

Test if your key supports the Elevation API by trying this link:
<https://maps.googleapis.com/maps/api/elevation/json?locations=0%2C0&key=YOUR_API_KEY>

### Dependencies

The library uses `trimesh` to read the GLB tiles and `DracoPy` to decompress the mesh compression used by Google's 3D Tiles.
Both are bundled in the release for **Linux** and **Windows**, so no extra setup is needed there.

> [!NOTE]
> The dependencies `numpy`, `Pillow` and `requests` are shipped with Houdini and are not bundled.

## Usage

Place the **Earth Cache** node in your scene, set the **Google API Key** and the **Coordinates** of the area you want to capture.

- **Shape**: select the region that selects tiles around coordinates.
- **Falloff**: blend from the region edge to the coarsest level over distance.
- **Error**: set the level of detail inside the region (`2-60000`).
- **Extract Textures**: write tile textures next to the cached GLB files.
- **Attributes**: each attribute stores the tile id of the tile at that geometric error level.
- **Max Meshes**: limit the number of downloaded tiles.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the Houdini development package and the release process.

## License

Copyright (c) 2026 Beat Reichenbach.
This project is licensed under the [GPLv3 License](LICENSE).

## Google Data

The geometry and imagery this tool produces come from Google's Photorealistic 3D Tiles and remain the copyright of Google and its licensors.
You are not allowed to use, redistribute or publish this data except as permitted by the [Google Maps Platform Terms of Service](https://cloud.google.com/maps-platform/terms).
