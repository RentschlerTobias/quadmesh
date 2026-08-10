#!/usr/bin/env python3
"""3D Hexa interpolation between Hub Master and Shroud Slave.

Reads hub_master.json, projects block corners onto Shroud radially,
performs TFI interpolation in the radial direction (Hub->Shroud),
exports VTK hexahedral blocks.

Usage:
    python hexa_interpolate.py [tag]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import meshio
from dp3d import unwrap_surface as us


def load_hub_master(path):
    with open(path) as f:
        return json.load(f)


def build_hexa_block(hub_corners, shroud_corners, n_r=4):
    """Build a 3D hexa block from Hub and Shroud quads.

    hub_corners: list of 4 [x,y,z] corners on Hub
    shroud_corners: list of 4 [x,y,z] corners on Shroud
    n_r: number of radial layers (n_r+1 points)

    Returns: (n_r+1, 4, 3) array of block vertices
    """
    hub = np.asarray(hub_corners, float)
    shroud = np.asarray(shroud_corners, float)
    assert hub.shape == (4, 3), f"Expected (4,3), got {hub.shape}"
    assert shroud.shape == (4, 3), f"Expected (4,3), got {shroud.shape}"

    t = np.linspace(0, 1, n_r + 1)[:, None, None]
    block = (1 - t) * hub[None, :, :] + t * shroud[None, :, :]
    return block


def export_vtk(hexa_blocks, out_path):
    """Export hexahedral blocks to VTK.

    hexa_blocks: list of (n_r+1, 4, 3) arrays
    """
    all_points = []
    cells = []
    offset = 0
    for block in hexa_blocks:
        n_layers = block.shape[0] - 1
        for i in range(n_layers):
            bottom = block[i]
            top = block[i + 1]
            hexa_points = np.vstack([bottom, top])
            all_points.append(hexa_points)
            cells.append(list(range(offset, offset + 8)))
            offset += 8

    if not all_points:
        print("[hexa] no blocks to export")
        return

    points = np.vstack(all_points)
    cell_array = np.array(cells)
    
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", cell_array)],
    )
    mesh.write(out_path, file_format="vtk", binary=False)
    print(f"[hexa] wrote {out_path} (ASCII VTK)")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ta"
    root = Path(__file__).resolve().parent.parent.parent

    hub_path = root / "output" / "T1_9" / "hub_stage1" / "tmesh" / "master" / f"hub_master_{tag}.json"
    hub_data = load_hub_master(hub_path)

    r_hub = hub_data["geometry"]["r"]
    blocks = hub_data["blocks"]

    shroud_stl = str(root / "T1_9_shroud_raw.stl")
    shroud_mesh = us.unwrap(shroud_stl)
    r_shroud = float(shroud_mesh["r"])

    print(f"[hexa] Hub r={r_hub:.3f}, Shroud r={r_shroud:.3f}")
    print(f"[hexa] {len(blocks)} Hub blocks")

    hexa_blocks = []
    for blk in blocks:
        if "corners_st" not in blk:
            continue
        hub_corners_3d = []
        shroud_corners_3d = []
        for c_st in blk["corners_st"]:
            s, t = c_st
            theta = s / r_hub if r_hub > 0 else 0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            hub_corners_3d.append([r_hub * cos_t, r_hub * sin_t, t])
            shroud_corners_3d.append([r_shroud * cos_t, r_shroud * sin_t, t])

        if len(hub_corners_3d) == 4 and len(shroud_corners_3d) == 4:
            hexa = build_hexa_block(hub_corners_3d, shroud_corners_3d, n_r=4)
            hexa_blocks.append(hexa)

    out_dir = root / "output" / "T1_9" / "hexa_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hexa_blocks_{tag}.vtk"
    export_vtk(hexa_blocks, out_path)

    print(f"[hexa] exported {len(hexa_blocks)} hexa blocks")


if __name__ == "__main__":
    main()
