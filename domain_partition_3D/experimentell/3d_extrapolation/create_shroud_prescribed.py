#!/usr/bin/env python3
"""Create prescribed singularities for Shroud based on blade geometry.

Instead of projecting Hub singularities (which collide on Shroud due to different
geometry), we place 4 singularities strategically around the Shroud blade:
- 2x index -1 at blade leading/trailing edges
- 2x index +1 on blade sides (suction/pressure)

This mirrors the Hub topology but adapts to Shroud geometry."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dp3d import unwrap_surface as us


def create_shroud_prescribed(stl_path, out_path):
    """Create prescribed singularity spec for Shroud."""
    mesh = us.unwrap(stl_path)
    
    r = float(mesh["r"])
    st = mesh["st"]
    blade_loops = mesh["blade_loops"]
    
    print(f"[shroud_prescribed] radius={r:.3f}, n_nodes={len(st)}")
    print(f"[shroud_prescribed] blade_loops: {len(blade_loops)}")
    
    if not blade_loops:
        print("[shroud_prescribed] ERROR: no blade loops found")
        return None
    
    # Use first blade loop
    loop = np.asarray(blade_loops[0], float)
    print(f"[shroud_prescribed] blade loop: {len(loop)} points")
    print(f"  s-range: [{loop[:,0].min():.3f}, {loop[:,0].max():.3f}]")
    print(f"  t-range: [{loop[:,1].min():.3f}, {loop[:,1].max():.3f}]")
    
    # Find leading edge (minimum s) and trailing edge (maximum s)
    le_idx = int(np.argmin(loop[:, 0]))
    te_idx = int(np.argmax(loop[:, 0]))
    
    le = loop[le_idx]
    te = loop[te_idx]
    
    print(f"[shroud_prescribed] LE: s={le[0]:.3f}, t={le[1]:.3f}")
    print(f"[shroud_prescribed] TE: s={te[0]:.3f}, t={te[1]:.3f}")
    
    # Compute blade center and normal
    center = loop.mean(axis=0)
    
    # Find suction side (upper t) and pressure side (lower t)
    # Split loop at LE and TE
    if le_idx < te_idx:
        upper = loop[le_idx:te_idx+1]
        lower = np.vstack([loop[te_idx:], loop[:le_idx+1]])
    else:
        upper = np.vstack([loop[le_idx:], loop[:te_idx+1]])
        lower = loop[te_idx:le_idx+1]
    
    # Suction side = higher t values on average
    if np.mean(upper[:, 1]) > np.mean(lower[:, 1]):
        suction = upper
        pressure = lower
    else:
        suction = lower
        pressure = upper
    
    # Find midpoints of suction and pressure sides
    suction_mid = suction[len(suction)//2]
    pressure_mid = pressure[len(pressure)//2]
    
    print(f"[shroud_prescribed] suction mid: s={suction_mid[0]:.3f}, t={suction_mid[1]:.3f}")
    print(f"[shroud_prescribed] pressure mid: s={pressure_mid[0]:.3f}, t={pressure_mid[1]:.3f}")
    
    # Place singularities offset from blade surface
    offset = 0.15  # larger offset for better block formation
    
    # Compute outward normal (away from blade interior)
    # Use loop tangent and perpendicular
    tangent = te - le
    normal = np.array([-tangent[1], tangent[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    
    prescribed = []
    
    # 2x -1 singularities at LE and TE (outside blade)
    for i, (pos, name) in enumerate([(le, "LE"), (te, "TE")]):
        sing_pos = pos - offset * normal  # offset in direction away from blade
        prescribed.append({
            "id": f"sing_{i}",
            "index": -1,
            "separatrix_count": 5,
            "position_st": sing_pos.tolist(),
            "position_st_3d": [r * np.cos(sing_pos[0]/r), r * np.sin(sing_pos[0]/r), sing_pos[1]],
            "blade_parametric": {
                "location": name,
                "offset": offset,
            }
        })
    
    # 2x +1 singularities on suction and pressure sides
    for i, (pos, name) in enumerate([(suction_mid, "suction"), (pressure_mid, "pressure")]):
        sing_pos = pos + offset * normal  # offset in opposite direction
        prescribed.append({
            "id": f"sing_{i+2}",
            "index": 1,
            "separatrix_count": 3,
            "position_st": sing_pos.tolist(),
            "position_st_3d": [r * np.cos(sing_pos[0]/r), r * np.sin(sing_pos[0]/r), sing_pos[1]],
            "blade_parametric": {
                "location": name,
                "offset": offset,
            }
        })
    
    data = {
        "schema_version": "1.0.0",
        "case_id": "T1_9",
        "surface": "shroud",
        "tag": "ta",
        "geometry": {
            "r": r,
            "pitch": mesh.get("pitch", 0),
            "blade_count": 8,
            "periodic": True,
        },
        "singularities": prescribed,
        "invariants": {
            "total_singularities": len(prescribed),
            "index_sum": sum(s["index"] for s in prescribed),
        }
    }
    
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"[shroud_prescribed] wrote {out_path}")
    print(f"[shroud_prescribed] {len(prescribed)} singularities, index sum = {data['invariants']['index_sum']}")
    
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        stl = "T1_9_shroud_raw.stl"
        out = "output/T1_9/shroud_stage1/shroud_prescribed_ta.json"
    else:
        stl = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "shroud_prescribed.json"
    
    create_shroud_prescribed(stl, out)
