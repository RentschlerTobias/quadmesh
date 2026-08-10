#!/usr/bin/env python3
"""Cross-field solver with enforced singularities for Shroud.

Implements a pragmatic approach to modify an existing cross-field to enforce
desired singularity positions and indices. Based on Jezdimirovic 2022 two-step
method: non-linear GL basis field + prescribed singularities via local
modifications.

Usage:
    python enforce_singularities.py <mesh_path> <singularity_spec.json> <output_path>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dp3d.dp_adapter import build_dp_data
from dp3d.field import FrameField
from dp3d.field.singularity_detector import detect_singularities


def solve_field_with_singularities(mesh, prescribed_singularities, n_iters=50):
    """Modify cross-field to enforce prescribed singularities.
    
    Args:
        mesh: PyG Data object with cross-field (mesh.u)
        prescribed_singularities: list of dicts with keys:
            position_st: [s, t] coordinates
            index: +1 or -1
        n_iters: number of smoothing iterations
    
    Returns:
        Modified mesh with enforced singularities
    """
    # Start from current field
    u = mesh.u.clone()
    nodes = mesh.x[:, 0:2].numpy()
    faces = mesh.faces.T.numpy()
    
    # Convert prescribed positions to normalized coordinates
    smin, smax = nodes[:, 0].min(), nodes[:, 0].max()
    tmin, tmax = nodes[:, 1].min(), nodes[:, 1].max()
    
    # Find face centers
    centroids = nodes[faces].mean(axis=1)
    
    # Identify target faces for singularities
    target_faces = {}
    for sing in prescribed_singularities:
        pos = np.asarray(sing["position_st"], float)
        # Normalize to mesh coordinates
        pos_norm = np.array([
            (pos[0] - smin) / (smax - smin),
            (pos[1] - tmin) / (tmax - tmin)
        ])
        
        # Find nearest face
        d2 = np.sum((centroids - pos_norm) ** 2, axis=1)
        face_id = int(np.argmin(d2))
        target_faces[face_id] = {
            "index": int(sing["index"]),
            "position": pos_norm,
            "original_pos": pos,
        }
    
    print(f"[enforce] targeting {len(target_faces)} faces for singularity enforcement")
    
    # Iterative smoothing with singularity constraints
    for iteration in range(n_iters):
        u_new = u.clone()
        
        # Smoothing: average with neighbors
        for i in range(len(u)):
            # Find neighboring nodes
            nbrs = []
            for fi in range(len(faces)):
                if i in faces[fi]:
                    for j in faces[fi]:
                        if j != i:
                            nbrs.append(j)
            
            if nbrs:
                # Average neighbor fields (with periodic angle handling)
                nbr_u = u[nbrs]
                # Align angles
                ref_angle = torch.atan2(u[i, 1], u[i, 0])
                aligned = []
                for nu in nbr_u:
                    angle = torch.atan2(nu[1], nu[0])
                    diff = angle - ref_angle
                    # Normalize to [-pi, pi]
                    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
                    aligned_angle = ref_angle + diff
                    aligned.append(torch.stack([
                        torch.cos(aligned_angle),
                        torch.sin(aligned_angle)
                    ]))
                
                avg = torch.stack(aligned).mean(dim=0)
                avg = avg / (torch.norm(avg) + 1e-8)
                
                # Blend with current
                alpha = 0.5
                u_new[i] = alpha * u[i] + (1 - alpha) * avg
                u_new[i] = u_new[i] / (torch.norm(u_new[i]) + 1e-8)
        
        # Enforce singularity constraints at target faces
        for face_id, target in target_faces.items():
            face_nodes = faces[face_id]
            idx = target["index"]
            
            # For a +1 singularity: field rotates +90° going around
            # For a -1 singularity: field rotates -90° going around
            # We enforce this by setting the face field to a specific pattern
            
            # Get face centroid
            centroid = nodes[face_nodes].mean(axis=0)
            
            # Set field at face nodes to create the singularity
            for j, nid in enumerate(face_nodes):
                # Direction from centroid to node
                to_node = nodes[nid] - centroid
                to_node = to_node / (np.linalg.norm(to_node) + 1e-8)
                
                # Rotate by +90° * index to create singularity
                angle = np.arctan2(to_node[1], to_node[0])
                sing_angle = angle + idx * np.pi / 2
                
                u_new[nid] = torch.tensor([
                    np.cos(sing_angle),
                    np.sin(sing_angle)
                ], dtype=torch.float)
        
        u = u_new
    
    mesh.u = u
    return mesh


def create_prescribed_from_hub_template(hub_master_path, shroud_stl_path):
    """Create prescribed singularity spec from Hub template scaled to Shroud."""
    import unwrap_surface as us
    
    hub_data = json.loads(Path(hub_master_path).read_text())
    shroud_mesh = us.unwrap(shroud_stl_path)
    
    r_hub = hub_data["geometry"]["r"]
    r_shroud = float(shroud_mesh["r"])
    ratio = r_shroud / r_hub
    
    prescribed = []
    for sing in hub_data["singularities"]:
        pos_hub = np.asarray(sing["position_st"], float)
        pos_shroud = pos_hub * ratio
        
        prescribed.append({
            "id": sing["id"],
            "index": sing["index"],
            "position_st": pos_shroud.tolist(),
            "source": "hub_template",
        })
    
    return prescribed


def main():
    if len(sys.argv) < 4:
        print("Usage: python enforce_singularities.py <shroud_stl> <hub_master.json> <output_dir>")
        sys.exit(1)
    
    shroud_stl = sys.argv[1]
    hub_master = sys.argv[2]
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create prescribed singularities
    prescribed = create_prescribed_from_hub_template(hub_master, shroud_stl)
    
    print(f"[enforce] {len(prescribed)} prescribed singularities from Hub template")
    for p in prescribed:
        print(f"  {p['id']}: index={p['index']:+d}, pos=({p['position_st'][0]:.3f}, {p['position_st'][1]:.3f})")
    
    # Build mesh
    mesh, transform = build_dp_data(shroud_stl)
    
    # Initialize field
    ff = FrameField(mesh)
    
    # Enforce singularities
    mesh = solve_field_with_singularities(mesh, prescribed, n_iters=100)
    
    # Detect singularities in modified field
    m = detect_singularities(ff.mesh)
    n_sing = (m.singularities != 0).sum().item()
    print(f"\n[enforce] After enforcement: {n_sing} singularities detected")
    print(f"Indices: {m.singularities[m.singularities != 0].tolist()}")
    
    # Export modified field
    out_path = out_dir / "shroud_enforced_field.pt"
    torch.save({
        "u": mesh.u,
        "prescribed": prescribed,
        "transform": transform,
    }, out_path)
    print(f"[enforce] Exported modified field to {out_path}")


if __name__ == "__main__":
    main()
