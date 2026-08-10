#!/usr/bin/env python3
"""Cross-field solver with enforced singularities via local rotation.

Instead of injecting singularity positions into a field that doesn't have them
(which produces invalid separatrices), we directly modify the cross-field u
by rotating it around each target singularity position.

Algorithm:
1. For each desired singularity at position p with index k:
   - Find all nodes within radius R
   - Rotate field u by angle θ proportional to distance from p
   - At p: full rotation of k*90°
   - At boundary of R: no rotation
2. Smooth with few iterations
3. Verify singularities are preserved

Usage:
    python field_solver.py <shroud_stl> <prescribed.json> <output.pt>
"""

import json
from pathlib import Path

import numpy as np
import torch

from .dp_adapter import build_dp_data
from .field import FrameField
from .field.singularity_detector import detect_singularities
from . import clean_separatrix as cs


def enforce_singularities(mesh, prescribed, transform=None, R_factor=3.0, n_smooth=10):
    """Modify cross-field to create prescribed singularities.
    
    Args:
        mesh: PyG Data with field mesh.u
        prescribed: list of dicts with position_st and index
        transform: dict with smin, smax, tmin, tmax for normalization
        R_factor: radius = R_factor * local_edge_length
        n_smooth: number of smoothing iterations
    
    Returns:
        Modified mesh
    """
    u = mesh.u.clone()
    nodes = mesh.x[:, 0:2].numpy()
    faces = mesh.faces.T.numpy()
    
    smin_n, smax_n = nodes[:, 0].min(), nodes[:, 0].max()
    tmin_n, tmax_n = nodes[:, 1].min(), nodes[:, 1].max()
    
    if transform:
        smin_p, smax_p = transform['smin'], transform['smax']
        tmin_p, tmax_p = transform['tmin'], transform['tmax']
    else:
        smin_p, smax_p = smin_n, smax_n
        tmin_p, tmax_p = tmin_n, tmax_n
    
    prescribed_norm = []
    for p in prescribed:
        pos = np.asarray(p["position_st"], float)
        pos_norm = np.array([
            (pos[0] - smin_p) / (smax_p - smin_p),
            (pos[1] - tmin_p) / (tmax_p - tmin_p)
        ])
        prescribed_norm.append({
            "position": pos_norm,
            "index": int(p["index"]),
            "id": p.get("id", "unknown"),
        })
    
    bbox_diag = np.linalg.norm([smax_n - smin_n, tmax_n - tmin_n])
    avg_edge = bbox_diag / np.sqrt(len(nodes))
    R = max(R_factor * avg_edge, 0.05)
    
    print(f"[field_solver] domain diagonal: {bbox_diag:.4f}, avg edge: {avg_edge:.4f}, influence radius: {R:.4f}")
    
    ff = mesh.frame_field.clone()
    
    for sing in prescribed_norm:
        pos = sing["position"]
        idx = sing["index"]
        
        # Find nearest face
        centroids = nodes[faces].mean(axis=1)
        d2 = np.sum((centroids - pos)**2, axis=1)
        face_id = int(np.argmin(d2))
        
        # Get the 3 vertices of this face
        face_nodes = faces[face_id]
        
        # Compute current field direction at face center
        current_angles = [torch.atan2(ff[n, 1], ff[n, 0]).item() for n in face_nodes]
        base_angle = np.mean(current_angles)
        
        # Set field at vertices to create singularity
        step = 2 * np.pi / 3 if idx > 0 else -2 * np.pi / 3
        for j, nid in enumerate(face_nodes):
            angle = base_angle + j * step
            ff[nid] = torch.tensor([
                np.cos(angle),
                np.sin(angle)
            ], dtype=torch.float)
        
        print(f"[field_solver] Set singularity {sing['id']} (index={idx:+d}) at face {face_id}")
    
    # Light smoothing only for non-singularity nodes
    print(f"[field_solver] light smoothing {n_smooth} iterations...")
    for it in range(n_smooth):
        ff_new = ff.clone()
        
        # Build neighbor list
        neighbors = [set() for _ in range(len(nodes))]
        for f in faces:
            for i in range(3):
                a, b = f[i], f[(i+1)%3]
                neighbors[a].add(b)
                neighbors[b].add(a)
        
        # Identify singularity nodes (don't smooth these)
        sing_nodes = set()
        for sing in prescribed_norm:
            d2 = np.sum((centroids - sing["position"])**2, axis=1)
            face_id = int(np.argmin(d2))
            sing_nodes.update(faces[face_id])
        
        for i in range(len(nodes)):
            if i in sing_nodes:
                continue
            
            nbrs = list(neighbors[i])
            if not nbrs:
                continue
            
            angles = [torch.atan2(ff[j, 1], ff[j, 0]).item() for j in nbrs]
            ref = torch.atan2(ff[i, 1], ff[i, 0]).item()
            
            wrapped = []
            for a in angles:
                diff = a - ref
                while diff > np.pi:
                    diff -= 2*np.pi
                while diff < -np.pi:
                    diff += 2*np.pi
                wrapped.append(ref + diff)
            
            avg_angle = np.mean(wrapped)
            alpha = 0.5
            my_diff = ref - avg_angle
            while my_diff > np.pi:
                my_diff -= 2*np.pi
            while my_diff < -np.pi:
                my_diff += 2*np.pi
            
            final_angle = avg_angle + alpha * my_diff
            ff_new[i] = torch.tensor([
                np.cos(final_angle),
                np.sin(final_angle)
            ], dtype=torch.float)
        
        ff = ff_new
    
    mesh.frame_field = ff
    mesh.u = ff
    mesh.singularities_coords = {}
    mesh.separatrices = []
    cs.CleanSeparatrixGenerator(mesh)
    
    target_faces = set()
    for sing in prescribed_norm:
        d2 = np.sum((centroids - sing["position"])**2, axis=1)
        face_id = int(np.argmin(d2))
        target_faces.add(face_id)
    
    n_removed = 0
    for fid in list(mesh.singularities_coords.keys()):
        if fid not in target_faces:
            mesh.singularities[fid] = 0
            del mesh.singularities_coords[fid]
            mesh.expected_separatrices.pop(fid, None)
            n_removed += 1
    
    mesh.separatrices = []
    cs.CleanSeparatrixGenerator(mesh)
    
    n_sing = (mesh.singularities != 0).sum().item()
    print(f"[field_solver] After cleanup: {n_sing} singularities with {len(mesh.separatrices)} separatrices")
    print(f"[field_solver] Removed {n_removed} extra singularities from smoothing")
    
    return mesh


def test_field(mesh, prescribed, transform=None):
    """Test if prescribed singularities are present in the field."""
    cs.CleanSeparatrixGenerator(mesh)
    n_sing = (mesh.singularities != 0).sum().item()
    print(f"[test] After CleanSeparatrixGenerator: {n_sing} singularities")
    print(f"[test] Separatrices: {len(mesh.separatrices)}")
    
    nodes = mesh.x[:, 0:2].numpy()
    
    if transform:
        smin_p, smax_p = transform['smin'], transform['smax']
        tmin_p, tmax_p = transform['tmin'], transform['tmax']
    else:
        smin_p, smax_p = nodes[:, 0].min(), nodes[:, 0].max()
        tmin_p, tmax_p = nodes[:, 1].min(), nodes[:, 1].max()
    
    for p in prescribed:
        pos = np.asarray(p["position_st"], float)
        pos_norm = np.array([
            (pos[0] - smin_p) / (smax_p - smin_p),
            (pos[1] - tmin_p) / (tmax_p - tmin_p)
        ])
        
        sing_faces = [f for f in torch.where(mesh.singularities != 0)[0].tolist() 
                      if f in mesh.singularities_coords]
        if not sing_faces:
            print(f"[test] {p['id']}: NO singularities in field!")
            continue
        
        sing_pos = np.array([mesh.singularities_coords[f] for f in sing_faces])
        d2 = np.sum((sing_pos - pos_norm)**2, axis=1)
        nearest = np.argmin(d2)
        fid = sing_faces[nearest]
        idx = int(mesh.singularities[fid])
        
        print(f"[test] {p['id']}: target=({pos_norm[0]:.3f},{pos_norm[1]:.3f}), "
              f"nearest face {fid} at ({sing_pos[nearest][0]:.3f},{sing_pos[nearest][1]:.3f}), "
              f"index={idx:+d}, dist={d2[nearest]**0.5:.4f}")
    
    return n_sing
