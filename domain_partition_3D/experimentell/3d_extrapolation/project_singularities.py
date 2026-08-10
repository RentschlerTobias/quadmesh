#!/usr/bin/env python3
"""Project Hub Master singularities onto Shroud Slave surface.

Reads hub_master.json (exported by export_hub_master.py) and projects each
singularity's barycentric (s,d) coordinates onto the Shroud blade profile.
Produces shroud_prescribed.json with the Shroud singularity positions.

Usage:
    python project_singularities.py <hub_master.json> <shroud_stl> <output_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dp3d import unwrap_surface as us
from dp3d import partition_surface as ps


def load_hub_master(path):
    data = json.loads(Path(path).read_text())
    assert data["schema_version"] == "1.0.0", f"Unsupported schema: {data['schema_version']}"
    assert data["surface"] == "hub", "Expected hub surface"
    return data


def project_onto_shroud(hub_data, shroud_stl_path):
    """Project Hub singularities onto Shroud using barycentric (s,d).

    Args:
        hub_data: dict from hub_master.json
        shroud_stl_path: path to Shroud STL file

    Returns:
        dict with Shroud singularity positions and metadata
    """
    shroud_mesh = us.unwrap(shroud_stl_path)
    shroud_blade = shroud_mesh.get("blade_loops", [])

    r_hub = hub_data["geometry"]["r"]
    r_shroud = float(shroud_mesh["r"])
    radius_ratio = r_shroud / r_hub if r_hub > 0 else 1.0

    prescribed = []
    for sing in hub_data["singularities"]:
        bp = sing.get("blade_parametric")
        if bp:
            s = bp["s"]
            d_hub = bp["d"]
            d_shroud = d_hub * radius_ratio
            blade_loop_id = bp.get("blade_loop_id", 0)
            side = bp.get("side", "suction")

            if blade_loop_id >= len(shroud_blade):
                print(f"[warn] blade_loop_id {blade_loop_id} out of range, using 0")
                blade_loop_id = 0

            shroud_loop = np.asarray(shroud_blade[blade_loop_id], float)
            pos_shroud, dist_err = _project_on_loop(s, d_shroud, shroud_loop, side, r_shroud)

            prescribed.append({
                "id": sing["id"],
                "hub_node_id": sing["node_id"],
                "index": sing["index"],
                "separatrix_count": sing["separatrix_count"],
                "position_st": pos_shroud.tolist(),
                "position_st_3d": _st_to_3d(pos_shroud, r_shroud).tolist(),
                "blade_parametric": {
                    "s": s,
                    "d": d_shroud,
                    "d_hub": d_hub,
                    "radius_ratio": radius_ratio,
                    "blade_loop_id": blade_loop_id,
                    "side": side,
                },
                "projection_error": float(dist_err),
            })
        else:
            # Interior singularity not near blade: radially scale (s,t) coordinates
            pos_hub = np.asarray(sing["position_st"], float)
            pos_shroud = pos_hub * radius_ratio
            prescribed.append({
                "id": sing["id"],
                "hub_node_id": sing["node_id"],
                "index": sing["index"],
                "separatrix_count": sing["separatrix_count"],
                "position_st": pos_shroud.tolist(),
                "position_st_3d": _st_to_3d(pos_shroud, r_shroud).tolist(),
                "projection_error": 0.0,
            })

    return {
        "schema_version": "1.0.0",
        "case_id": hub_data["case_id"],
        "surface": "shroud",
        "tag": hub_data["tag"],
        "geometry": {
            "r": r_shroud,
            "pitch": hub_data["geometry"]["pitch"],
            "blade_count": hub_data["geometry"]["blade_count"],
            "periodic": hub_data["geometry"]["periodic"],
        },
        "singularities": prescribed,
        "invariants": {
            "total_singularities": len(prescribed),
            "index_sum": sum(s["index"] for s in prescribed),
        },
    }


def _project_on_loop(s, d, loop, side, r_shroud, tol=1e-6):
    """Find point on loop at arc-length s, then offset radially by distance d.

    Returns (position_st, distance_error).
    """
    loop = np.asarray(loop, float)
    if len(loop) < 2:
        return loop[0] if len(loop) > 0 else np.array([0.0, 0.0]), 0.0

    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:
        return loop[0], 0.0

    target = s * total
    seg_idx = np.clip(np.searchsorted(cum, target, side="right") - 1, 0, len(seg) - 1)
    base = np.array([np.interp(target, cum, loop[:, 0]),
                     np.interp(target, cum, loop[:, 1])])

    # Radial offset: move along the blade-loop normal. In (s,t) unwrapped
    # coordinates the cylinder radial direction is perpendicular to the blade
    # tangent, so the loop normal is a good proxy. Sign follows Hub convention:
    # suction = away from blade interior, pressure = toward it.
    tangent = loop[min(seg_idx + 1, len(loop) - 1)] - loop[seg_idx]
    normal = np.array([-tangent[1], tangent[0]])
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-12:
        return base, 0.0
    normal = normal / n_norm

    sign = 1.0 if side == "suction" else -1.0

    # Clamp d to avoid going outside mesh (heuristic: 30% of radius).
    d_clamped = min(d, r_shroud * 0.3)
    pos = base + sign * d_clamped * normal

    _, _, dist_back, _ = ps._project_to_polyline(pos, loop)
    dist_err = abs(dist_back - d_clamped)

    return pos, dist_err


def _st_to_3d(st, r):
    """Convert (s,t) unwrapped coordinates back to 3D (x,y,z)."""
    s, t = st
    theta = s / r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = t
    return np.array([x, y, z])


def main():
    if len(sys.argv) < 3:
        print("Usage: python project_singularities.py <hub_master.json> <shroud_stl> [output_dir]")
        sys.exit(1)

    hub_path = sys.argv[1]
    shroud_stl = sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output") / "T1_9" / "shroud_stage1"

    print(f"[project] loading Hub master from {hub_path}")
    hub_data = load_hub_master(hub_path)

    print(f"[project] projecting onto Shroud: {shroud_stl}")
    result = project_onto_shroud(hub_data, shroud_stl)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shroud_prescribed_{hub_data['tag']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[project] wrote {out_path}")
    print(f"[project] {len(result['singularities'])} singularities projected, "
          f"index sum = {result['invariants']['index_sum']}")


if __name__ == "__main__":
    main()
