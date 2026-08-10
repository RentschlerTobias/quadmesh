#!/usr/bin/env python3
"""Export Hub Master data for Shroud Slave pipeline.

Produces a structured export of the Hub partition (singularities, blade profile,
block topology, TFI grids) that serves as the Master contract for:
  - Phase 2: Shroud prescribed-singularity placement
  - Phase 3: 3D hexa interpolation Hub->Shroud

Output: output/T1_9/hub_stage1/tmesh/master/hub_master.json
        + companion .npz files for large arrays
"""

import json
from pathlib import Path

import numpy as np


def _to_json_safe(obj):
    """Recursively convert numpy/torch objects to JSON-safe Python types."""
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if obj is None:
        return None
    return obj


def export_hub_master(run_result, out_dir, tag="ta"):
    """Export the full Hub Master dataset from a successful run_tmesh() result.

    Args:
        run_result: dict returned by run_tmesh() containing:
            metrics, result, tfi, mesh, seam_info, boundary_ref,
            edge_samples, divisions, transform, seam_pairs
        out_dir: Path to output directory (e.g. output/T1_9/hub_stage1/tmesh/)
        tag: pipeline variant tag ("ta" or "tb")

    Returns:
        Path to the written hub_master.json file
    """
    out_dir = Path(out_dir)
    master_dir = out_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    mesh = run_result["mesh"]
    result = run_result["result"]
    tfi = run_result["tfi"]
    seam_info = run_result["seam_info"]
    transform = run_result["transform"]
    metrics = run_result["metrics"]

    geometry = {
        "r": float(transform.get("r", 0.0)),
        "pitch": float(mesh.pitch_norm),
        "s_range": [float(transform["smin"]), float(transform["smax"])],
        "t_range": [float(transform["tmin"]), float(transform["tmax"])],
        "blade_count": 8,  # T1_9 has 8 blades
        "periodic": True,
    }

    singularities = []
    if hasattr(mesh, "singularities_coords") and mesh.singularities_coords:
        for node_id, coord in mesh.singularities_coords.items():
            idx = int(mesh.singularities[node_id])
            # Kowalski Prop 9: idx+1 -> 3 separatrices, idx-1 -> 5 separatrices
            sep_count = 3 if idx > 0 else 5

            sing = {
                "id": f"sing_{len(singularities)}",
                "node_id": int(node_id),
                "index": idx,
                "position_st": _to_json_safe(coord),
                "separatrix_count": sep_count,
            }

            # Compute (s,d) barycentric projection along blade profile
            # s = position along blade loop [0,1], d = normal distance
            blade_param = _compute_blade_parametric(coord, mesh.blade_loops)
            if blade_param:
                sing["blade_parametric"] = blade_param
            else:
                # Interior singularities not near blade: store raw (s,t) for
                # radial projection to Shroud
                sing["position_st_raw"] = _to_json_safe(coord)

            singularities.append(sing)

    blocks = []
    nodes = result.get("nodes", np.zeros((0, 2)))
    for i, blk in enumerate(result.get("blocks", [])):
        cids = blk["corners"]
        corners_st = nodes[cids].tolist() if len(cids) > 0 and len(nodes) > 0 else []
        block = {
            "id": f"block_{i}",
            "corners": _to_json_safe(cids),
            "corners_st": _to_json_safe(corners_st),
            "corner_pos": _to_json_safe(blk.get("corner_pos", [])),
            "sides": [
                {
                    "from": int(blk["corners"][j]),
                    "to": int(blk["corners"][(j + 1) % 4]),
                    "type": _classify_side(blk, j, result),
                }
                for j in range(4)
            ],
            "side_chains": _to_json_safe(blk.get("side_chains", [])),
            "area": float(blk.get("area", 0.0)),
            "centroid": _to_json_safe(blk.get("centroid", [])),
        }
        blocks.append(block)

    index_sum = sum(s["index"] for s in singularities)
    invariants = {
        "total_singularities": len(singularities),
        "index_sum": index_sum,
        "block_count": len(blocks),
        "inverted_blocks": metrics.get("inverted_blocks", 0),
        "irregular_interior_nodes": metrics.get("irregular_interior_nodes", 0),
        "rejected_regions": metrics.get("rejected_regions", 0),
    }

    npz_path = master_dir / f"hub_master_{tag}.npz"
    if tfi and tfi.get("grids"):
        grids = tfi["grids"]
        np.savez(
            npz_path,
            grids=np.array(grids, dtype=object),
            seam_pts_L=np.array(tfi.get("seam_pts", {}).get("L", []), dtype=object),
            seam_pts_R=np.array(tfi.get("seam_pts", {}).get("R", []), dtype=object),
        )
    else:
        npz_path = None

    master = {
        "schema_version": "1.0.0",
        "case_id": "T1_9",
        "surface": "hub",
        "tag": tag,
        "exported_at": str(np.datetime64("now")),
        "geometry": geometry,
        "singularities": singularities,
        "blocks": blocks,
        "invariants": invariants,
        "metrics": metrics,
        "transform": _to_json_safe(transform),
        "companion_npz": str(npz_path.relative_to(master_dir)) if npz_path else None,
    }

    json_path = master_dir / f"hub_master_{tag}.json"
    json_path.write_text(json.dumps(master, indent=2, default=_to_json_safe))
    print(f"[export] wrote {json_path}")
    if npz_path:
        print(f"[export] wrote {npz_path}")

    return json_path


def _compute_blade_parametric(coord, blade_loops, tol=1e-6):
    """Compute barycentric (s,d) for a point relative to the blade profile.

    s = normalized arc-length along blade loop [0=LE, 1=TE]
    d = signed normal distance from blade (positive = away from blade interior)

    Returns dict with s, d, blade_loop_id, side or None if not near blade.
    """
    p = np.asarray(coord, float)
    for loop_id, loop in enumerate(blade_loops):
        loop = np.asarray(loop, float)
        if loop.ndim != 2 or len(loop) < 2:
            continue

        seg, t, dist, proj = _project_to_polyline(p, loop)
        if dist > 0.05:  # not near this blade
            continue

        # Compute s = normalized arc-length to projection point
        seg_lengths = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        s = float((cum_lengths[seg] + t * seg_lengths[seg]) / cum_lengths[-1])

        # Determine side: suction vs pressure
        # Use cross product of tangent and vector to point
        tangent = loop[min(seg + 1, len(loop) - 1)] - loop[seg]
        to_point = p - proj
        side = "suction" if np.cross(tangent, to_point) > 0 else "pressure"

        return {
            "s": round(s, 6),
            "d": round(float(dist), 6),
            "blade_loop_id": loop_id,
            "side": side,
            "projection_tolerance": tol,
        }
    return None


def _project_to_polyline(p, poly):
    """Project point p onto polyline. Returns (seg_idx, t, dist, proj_point)."""
    p = np.asarray(p, float)
    poly = np.asarray(poly, float)
    best = (0, 0.0, np.inf, poly[0])
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-18:
            continue
        t = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best[2]:
            best = (i, float(t), float(d), proj)
    return best


def _classify_side(blk, side_idx, result):
    """Classify a block side type: separatrix, boundary, seam, or blade."""
    corners = blk["corners"]
    c1, c2 = corners[side_idx], corners[(side_idx + 1) % 4]

    # Check if side is along blade boundary
    if result.get("blade_regions"):
        blade_nodes = set()
        for reg in result["blade_regions"]:
            blade_nodes.update(reg)
        if c1 in blade_nodes and c2 in blade_nodes:
            return "blade"

    # Check if side connects to a singularity
    if hasattr(result.get("mesh"), "singularities"):
        sing_nodes = set(np.where(result["mesh"].singularities != 0)[0])
        if c1 in sing_nodes or c2 in sing_nodes:
            return "separatrix"

    return "boundary"

