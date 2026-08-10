#!/usr/bin/env python3
"""
Stage 1 / Step B: Adapt an unwrapped 2D surface mesh into the
torch_geometric ``Data`` object that domain_partition's pipeline expects.

domain_partition's ``MeshGenerator`` produces a ``Data`` with:
    x            (N,3)  [x_norm, y_norm, node_dim]   node_dim: 2=interior,1=edge,0=corner
    faces        (3,M)  triangle connectivity
    edge_index   (2,E)  undirected edges
    edge_attr    (E,)   1 = boundary edge
    streamlines  list   ordered boundary polylines (corner->corner), normalized coords
    nodes_faces_ids, face_attr, centerPoints

We reuse MeshGenerator's helper methods verbatim (via an __init__-bypassing shim)
so the downstream FrameField / StreamlineGenerator behave identically to the
native 2D case. Coordinates are normalized to [0,1]^2; the affine transform is
returned so blocks can be mapped back to (s,t) and then onto the 3D cylinder.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from .field.mesh_generator import MeshGenerator
from .unwrap_surface import unwrap


def _split_loop_at_corners(loop, corner_set):
    """Split a cyclic node loop into corner->corner segments (inclusive ends)."""
    n = len(loop)
    corner_pos = [i for i, v in enumerate(loop) if v in corner_set]
    if len(corner_pos) < 2:
        return [loop + [loop[0]]]  # degenerate: whole closed loop
    segments = []
    for k in range(len(corner_pos)):
        i0 = corner_pos[k]
        i1 = corner_pos[(k + 1) % len(corner_pos)]
        seg = []
        i = i0
        while True:
            seg.append(loop[i])
            if i == i1:
                break
            i = (i + 1) % n
        segments.append(seg)
    return segments


def build_dp_data(stl_path, corner_angle_deg=40.0):
    u = unwrap(stl_path, corner_angle_deg=corner_angle_deg)
    st = u["st"].astype(np.float64)
    tris = u["tris"]
    node_dim = u["node_dim"]

    # --- normalize to [0,1]^2 (record affine transform for inverse) ---
    smin, smax = st[:, 0].min(), st[:, 0].max()
    tmin, tmax = st[:, 1].min(), st[:, 1].max()
    st_norm = np.empty_like(st)
    st_norm[:, 0] = (st[:, 0] - smin) / (smax - smin)
    st_norm[:, 1] = (st[:, 1] - tmin) / (tmax - tmin)
    transform = {"smin": float(smin), "smax": float(smax),
                 "tmin": float(tmin), "tmax": float(tmax), "r": u["r"]}

    # --- core tensors ---
    x = torch.zeros((len(st), 3), dtype=torch.float)
    x[:, 0] = torch.from_numpy(st_norm[:, 0]).float()
    x[:, 1] = torch.from_numpy(st_norm[:, 1]).float()
    x[:, 2] = torch.from_numpy(node_dim).float()
    faces = torch.from_numpy(tris.T.astype(np.int64))

    shim = MeshGenerator.__new__(MeshGenerator)
    shim.is_quad_mesh = False
    edge_index = shim.face_to_edges(faces)
    mesh = Data(x=x, edge_index=edge_index, faces=faces)
    shim.mesh = mesh

    # --- boundary streamlines: corner->corner segments, normalized coords ---
    streamlines = []
    for lp, corners in zip(u["loops"], u["loop_corners"]):
        for seg in _split_loop_at_corners(lp, set(corners)):
            streamlines.append(st_norm[seg].astype(np.float64))
    mesh.streamlines = streamlines

    # --- reuse MeshGenerator helpers verbatim ---
    mesh.nodes_faces_ids = shim.map_nodes_to_faces()
    mesh.edge_attr = shim.add_edge_attr()
    mesh.face_attr = shim.add_face_attr()
    shim.getFaceCenterPoints()  # sets mesh.centerPoints

    # --- corner types + blade outlines (normalized to [0,1]^2 like mesh.x) ---
    mesh.corner_type = torch.from_numpy(u["corner_type"].astype(np.int64))

    def _normalize(arr):
        out = np.empty_like(arr, dtype=np.float64)
        out[:, 0] = (arr[:, 0] - smin) / (smax - smin)
        out[:, 1] = (arr[:, 1] - tmin) / (tmax - tmin)
        return out

    mesh.blade_loops = [_normalize(np.asarray(bl, float)) for bl in u["blade_loops"]]

    # pitchwise-periodic theta-boundary node pairs (master_left, slave_right).
    # Node ids index directly into mesh.x (build_dp_data does not renumber).
    pp = u.get("periodic_pairs", [])
    mesh.periodic_pairs = (torch.tensor(pp, dtype=torch.long) if pp
                           else torch.zeros((0, 2), dtype=torch.long))
    mesh.pitch = u.get("pitch")
    # pitch in NORMALIZED s (same linear s-scale as mesh.x[:,0]); the theta
    # periodicity is a pure s-translation, so a physical +pitch shift maps to a
    # +pitch_norm shift in normalized s (t unchanged). Used by the periodic block
    # tiling in partition_surface to replicate streamlines across the seam.
    mesh.pitch_norm = (float(float(mesh.pitch) / (smax - smin))
                       if mesh.pitch is not None else None)

    return mesh, transform


