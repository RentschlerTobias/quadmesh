"""Pure Xiao 2020 method: field/separatrix pipeline up to the endpoint snap,
then T-mesh block extraction -- no T-a/T-b seam postprocessing, no TFI fill.
Used as the baseline the Ansatz-T variants are compared against."""

import json
import os
import time
from pathlib import Path

import numpy as np

from . import partition_surface as ps
from . import tmesh_faces as tmf
from . import plotting
from .dp_adapter import build_dp_data
from .field import FrameField, StreamlineGenerator_v2
from .field.singularity_detector import detect_singularities
from .field.streamline_merging import StreamlineMerging
from .field.streamline_intersection_splitter import StreamlineIntersectionSplitter

# partition_surface monkey-patches StreamlineMerging.__init__ to run
# find_missed_streamline_endpoints (Xiao 2020 Alg. 2 Case 2). Without it the
# merge conflicts are not caught and the O-grid ring never closes.
assert hasattr(StreamlineMerging.__init__, "__wrapped__") or \
    "find_missed_streamline_endpoints" in \
    StreamlineMerging.__init__.__code__.co_names, \
    "StreamlineMerging patch missing -- partition_surface must be imported"


def run_xiao(stl, flat_tol_deg=15.0, verbose=True):
    ps.set_periodic(True)
    ps.set_tile_periodic(False)
    t0 = time.time()
    mesh, _tf = build_dp_data(stl)
    ff = FrameField(mesh)
    m = detect_singularities(ff.mesh)
    n_sing = int((m.singularities != 0).sum())
    sl = StreamlineGenerator_v2(ff.mesh)
    ps._drop_degenerate_corner_seps(sl.mesh)
    ps._snap_separatrix_endpoints(sl.mesh, radius=0.045)

    merging = StreamlineMerging(sl.mesh, verbose=bool(os.environ.get("DP3D_DEBUG")))
    splitter = StreamlineIntersectionSplitter(offset_boundingBox=0.05,
                                              num_samples=5)
    updated = splitter.process_streamlines(merging.new_streamlines)
    n_boundary = len(sl.mesh.streamlines) - len(sl.mesh.separatrices)
    boundary_ref = [np.asarray(s, float)
                    for s in sl.mesh.streamlines[:n_boundary]]
    _bdist = lambda p: ps._min_boundary_dist(p, boundary_ref)
    gen = tmf.TMeshFaceGenerator(updated, blade_loops=list(mesh.blade_loops),
                                 flat_tol_deg=flat_tol_deg, verbose=verbose,
                                 boundary_dist_fn=_bdist, bnd_tol=1e-3)
    result = gen.get_blocks()

    _reg, tnodes, irregular = tmf.node_regularity(
        result, _bdist, classify_bnd_tol=1e-3)
    metrics = {
        "approach": "Xiao + custom clipping (no seam postprocessing)",
        "singularities": n_sing,
        "blocks": len(result["blocks"]),
        "nonquad_regions": len(result["rejects"]),
        "irregular_interior_nodes": len(irregular),
        "runtime_s": round(time.time() - t0, 1),
    }
    return {"metrics": metrics, "result": result, "mesh": mesh,
            "sl": sl, "irregular": irregular, "tnodes": tnodes,
            "merging": merging, "boundary_ref": boundary_ref}


def write_xiao_report(data, out_dir):
    """Streamline/singularity text tables and metrics JSON for a run_xiao
    result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sl = data["sl"]
    metrics = data["metrics"]
    singularities = plotting.get_singularities(sl.mesh)
    connections = plotting.singularity_connections(sl.mesh, singularities)
    n_b = len(sl.mesh.streamlines) - len(sl.mesh.separatrices)

    lines = ["Streamline index table", "=" * 60, "",
             f"BOUNDARY streamlines (indices 0 .. {n_b - 1}):"]
    for i in range(n_b):
        s = np.asarray(sl.mesh.streamlines[i], float)
        lines.append(f"  {i:3d}:  len={len(s):4d}  "
                     f"start=({s[0, 0]:.4f},{s[0, 1]:.4f})  "
                     f"end=({s[-1, 0]:.4f},{s[-1, 1]:.4f})")
    lines.append("")
    lines.append(f"SEPARATRICES (indices {n_b} .. "
                 f"{len(sl.mesh.streamlines) - 1}):")
    for i, d in enumerate(sl.mesh.separatrices):
        idx = n_b + i
        s = np.asarray(sl.mesh.streamlines[idx], float)
        sc = d.get("singularity_coords")
        sing_str = (f"S=({sc[0]:.4f},{sc[1]:.4f})" if sc is not None
                    else "S=?")
        lines.append(f"  {idx:3d}:  len={len(s):4d}  {sing_str}  "
                     f"start=({s[0, 0]:.4f},{s[0, 1]:.4f})  "
                     f"end=({s[-1, 0]:.4f},{s[-1, 1]:.4f})")
    (out_dir / "xiao_streamline_table.txt").write_text("\n".join(lines))

    lines = ["Singularity analysis", "=" * 60, ""]
    for name, d in singularities.items():
        c = d["coords"]
        lines.append(f"{name}: ({c[0]:.4f}, {c[1]:.4f})")
        lines.append(f"  Separatrices ({len(d['sep_indices'])} total):")
        for idx, tgt in d["sep_targets"]:
            tgt_name = plotting.find_target_singularity(tgt, singularities)
            if tgt_name:
                lines.append(f"    sl {idx}: -> {tgt_name} (direct sing-sing)")
            else:
                lines.append(f"    sl {idx}: -> boundary "
                             f"({tgt[0]:.4f},{tgt[1]:.4f})")
        lines.append("")
    lines.append("Singularity adjacency (direct connections):")
    names = [f"S{i}" for i in range(1, len(singularities) + 1)]
    for a in names:
        for b in names:
            if a >= b:
                continue
            key = tuple(sorted([a, b]))
            if key in connections:
                lines.append(f"  {a} - {b}: YES "
                             f"({len(connections[key])} streamlines)")
                for idx in connections[key]:
                    s = np.asarray(sl.mesh.streamlines[idx], float)
                    lines.append(f"      sl {idx}: len={len(s)}")
            else:
                lines.append(f"  {a} - {b}: NO DIRECT CONNECTION")
        lines.append("")
    (out_dir / "xiao_singularity_analysis.txt").write_text("\n".join(lines))

    (out_dir / "xiao_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote xiao report to {out_dir}")
