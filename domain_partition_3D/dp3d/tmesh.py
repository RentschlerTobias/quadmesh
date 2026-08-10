"""Ansatz T: periodic seam as WALL + master-slave junctions + T-mesh blocks + TFI.

Alternative to the wrap approach (partition_surface.partition with periodicity
on): streamlines END on the theta seam instead of integrating across it. The
cross FIELD stays periodic (seam weld), so the separatrix geometry mirrors mod
pitch by itself; only the block stage treats the seam as a wall:

  1. field/separatrix pipeline with set_periodic(True) + set_tile_periodic(False)
  2. seam symmetrization: junction sets of both seams are unioned mod pitch
     (DLR Sauer/Morsbach 2023 sec 2.7 master-slave: only the left seam is
     parametrized, the right seam is the exact +pitch translate). Mirrored
     junctions without an interior curve are hanging T-nodes -- optionally
     (variant T-b) a block edge is CONTINUED into the domain from every
     hanging junction (periodic continuation of the docking curve).
  3. block extraction tolerating T-junctions (tmesh_faces: a block needs
     exactly 4 REAL corners; flat ~180deg nodes are allowed on its sides)
  4. TFI (Coons) fill per block + Thomas-Middlecoff elliptic smoothing.
     Cell counts per graph edge come from an integer program (opposite block
     sides must carry the same number of cells; seam edge pairs share their
     count), every edge is sampled ONCE (tanh-clustered towards the blade
     boundary layer) and both adjacent blocks reference that sampling ->
     grid points are identical across every block edge (CFD-conforming),
     hanging T-nodes become regular grid points.
"""

import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import partition_surface as ps
from . import tmesh_faces as tmf
from .dp_adapter import build_dp_data
from .field import FrameField, StreamlineGenerator_v2
from .field.singularity_detector import detect_singularities
from .field.streamline_merging import StreamlineMerging
from .field.streamline_intersection_splitter import StreamlineIntersectionSplitter

H_CELL = 0.04            # target cell size (uniform reference)
CLUSTER_RATIO = 5.0      # generic wall: first cell ~ uniform/5 (tanh stretching)
BLADE_CLUSTER_RATIO = 50.0   # blade boundary layer (CFD): first cell ~ uniform/50
BLADE_NORMAL_CELLS = 12  # min cells across a blade-normal edge (BL resolution)


def _inject_prescribed_singularities(mesh, prescribed):
    """Replace auto-detected singularities with prescribed positions.

    prescribed: list of dicts with keys:
        position_st: [s, t] coordinates in unwrapped domain
        index: Poincaré index (+1 or -1)
        separatrix_count: 3 or 5
    """
    import torch

    nodes = mesh.x[:, 0:2].numpy()
    faces = mesh.faces.T.numpy()

    mesh.singularities[:] = 0
    mesh.singularities_coords = {}
    mesh.expected_separatrices = {}

    # Compute normalization from mesh bounds to match mesh.x [0,1] coordinates
    smin, smax = nodes[:, 0].min(), nodes[:, 0].max()
    tmin, tmax = nodes[:, 1].min(), nodes[:, 1].max()

    for p in prescribed:
        pos = np.asarray(p["position_st"], float)
        idx = int(p["index"])
        n_sep = int(p.get("separatrix_count", 5 if idx < 0 else 3))

        # Normalize physical (s,t) to [0,1] to match mesh.x coordinates
        pos_norm = np.array([
            (pos[0] - smin) / (smax - smin),
            (pos[1] - tmin) / (tmax - tmin)
        ])

        centroids = nodes[faces].mean(axis=1)
        d2 = np.sum((centroids - pos_norm) ** 2, axis=1)
        face_id = int(np.argmin(d2))

        mesh.singularities[face_id] = idx
        mesh.singularities_coords[face_id] = torch.tensor(pos_norm, dtype=torch.float)
        mesh.expected_separatrices[face_id] = n_sep

    print(f"[prescribed] injected {len(prescribed)} singularities into mesh")


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------

def _arclen(poly):
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _resample_at(poly, fracs):
    """Sample a polyline at the given arclength fractions (0..1)."""
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    if d[-1] < 1e-15:
        return np.repeat(poly[:1], len(fracs), axis=0)
    ss = np.asarray(fracs, float) * d[-1]
    return np.column_stack([np.interp(ss, d, poly[:, 0]),
                            np.interp(ss, d, poly[:, 1])])


def _dist_to_chain(p, chain):
    return ps._project_to_polyline(np.asarray(p, float), chain)[2]


# --------------------------------------------------------------------------
# start-kink repair (Xiao Case-2 dedup blend artifact)
# --------------------------------------------------------------------------

def fix_start_kinks(mesh, angle_deg=15.0, frac=0.25, max_len=0.12):
    """Straighten separatrix starts that leave their singularity in the wrong
    direction. The Case-2 geometric dedup in _snap_separatrix_endpoints blends
    a surviving curve with a near-duplicate from the OTHER cluster singularity
    (starts ~0.02 apart); the blended curve then bulges towards the victim's
    start before turning back (S-kink). Each separatrix dict stores the exact
    field emanation direction ("vector"); when the initial tangent deviates by
    more than angle_deg we morph the first part of the curve back onto the
    ray origin + a*vector (smooth quadratic weight, continuous at the far
    end)."""
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    fixed = 0
    for si, d in enumerate(mesh.separatrices):
        v = d.get("vector")
        o = d.get("singularity_coords")
        if v is None or o is None:
            continue
        v = np.asarray(v, float)
        v = v / (np.linalg.norm(v) + 1e-30)
        s = np.asarray(mesh.streamlines[n_b + si], float)
        if s.ndim != 2 or len(s) < 4:
            continue
        d0 = s[min(3, len(s) - 1)] - s[0]
        n0 = np.linalg.norm(d0)
        if n0 < 1e-12:
            continue
        ang = np.degrees(np.arccos(np.clip(float(d0 @ v) / n0, -1.0, 1.0)))
        if ang <= angle_deg:
            continue
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        a0 = min(frac * cum[-1], max_len)
        if a0 < 1e-9:
            continue
        w = np.clip(1.0 - cum / a0, 0.0, 1.0) ** 2
        ray = s[0][None, :] + cum[:, None] * v[None, :]
        mesh.streamlines[n_b + si] = w[:, None] * ray + (1 - w[:, None]) * s
        fixed += 1
    if fixed:
        print(f"[kink] straightened {fixed} separatrix start(s) onto their "
              f"emanation direction")


def resample_coarse_separatrices(mesh, n_min=12, seg_ratio=4.0, ds=0.004):
    """Uniformly resample separatrices with few or very unevenly spaced
    points. The intersection splitter fits an exact spline (splprep, s=0)
    through every curve; on a 4-point curve whose first segment is ~8x longer
    than the rest the spline parametrization overshoots and the straight
    curve comes back as an arc (arclen +40%, visible as a strongly bent block
    edge at the lower TE singularity). Uniform arclength sampling keeps
    straight polylines straight through the spline."""
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    fixed = 0
    for si in range(len(mesh.separatrices)):
        s = np.asarray(mesh.streamlines[n_b + si], float)
        if s.ndim != 2 or len(s) < 2:
            continue
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        seg = seg[seg > 1e-12]
        if len(seg) == 0:
            continue
        uneven = seg.max() / max(np.median(seg), 1e-12) > seg_ratio
        if len(s) >= n_min and not uneven:
            continue
        n = max(n_min, int(np.ceil(_arclen(s) / ds)) + 1)
        mesh.streamlines[n_b + si] = _resample_at(
            s, np.linspace(0.0, 1.0, n))
        fixed += 1
    if fixed:
        print(f"[resample] uniformly resampled {fixed} coarse "
              f"separatrix(es) (splitter spline overshoot guard)")


def straighten_sing_connectors(mesh, max_len=0.1):
    """Replace short singularity-singularity connectors by straight segments.

    The two integrations of a cluster connector (S0->S1 and S1->S0) are
    blended by the snap pass (Kowalski Eq 28); around a tip cluster the two
    paths disagree enough that the blend comes out S-shaped -- the curve first
    heads the wrong way, then turns (user finding). At cluster distance
    (~0.05) the field line is straight for all practical purposes, so the
    clean fix is geometric: straight connector, endpoints untouched."""
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    sings = np.array(list(mesh.singularities_coords.values())) \
        if getattr(mesh, "singularities_coords", None) else np.zeros((0, 2))
    fixed = 0
    for si, d in enumerate(mesh.separatrices):
        s = np.asarray(mesh.streamlines[n_b + si], float)
        if s.ndim != 2 or len(s) < 3 or _arclen(s) > max_len:
            continue
        if len(sings) == 0:
            break
        d0 = np.linalg.norm(sings - s[0], axis=1).min()
        d1 = np.linalg.norm(sings - s[-1], axis=1).min()
        if d0 < 0.02 and d1 < 0.02:
            n = max(len(s), 8)
            t = np.linspace(0.0, 1.0, n)[:, None]
            mesh.streamlines[n_b + si] = (1 - t) * s[0] + t * s[-1]
            fixed += 1
    if fixed:
        print(f"[kink] straightened {fixed} short sing-sing connector(s)")


# --------------------------------------------------------------------------
# seam identification + master-slave symmetrization
# --------------------------------------------------------------------------

def _periodic_chains(mesh):
    """(left, right) seam node chains from mesh.periodic_pairs, sorted by t."""
    pp = mesh.periodic_pairs
    pp = pp.numpy() if hasattr(pp, "numpy") else np.asarray(pp)
    X = mesh.x[:, 0:2].numpy()
    A, B = X[pp[:, 0]], X[pp[:, 1]]
    if A[:, 0].mean() > B[:, 0].mean():
        A, B = B, A
    return A[np.argsort(A[:, 1])], B[np.argsort(B[:, 1])]


def _chain_segments(segs):
    """Reassemble split wall segments into one polyline. Segment endpoints of a
    split wall coincide exactly, so exact-key chaining is safe."""
    from collections import Counter
    segs = [np.asarray(s, float) for s in segs]

    def key(p):
        return (round(float(p[0]), 9), round(float(p[1]), 9))

    cnt = Counter()
    for s in segs:
        cnt[key(s[0])] += 1
        cnt[key(s[-1])] += 1
    ends = [k for k, v in cnt.items() if v == 1]
    if len(ends) != 2:
        raise RuntimeError(f"wall segments do not chain (ends={len(ends)})")
    cur = min(ends, key=lambda k: k[1])          # start at smaller t
    remaining = list(range(len(segs)))
    parts = []
    while remaining:
        nxt = None
        for i in remaining:
            if key(segs[i][0]) == cur:
                nxt = (i, False)
                break
            if key(segs[i][-1]) == cur:
                nxt = (i, True)
                break
        if nxt is None:
            raise RuntimeError("wall segment chain broken")
        i, rev = nxt
        s = segs[i][::-1] if rev else segs[i]
        parts.append(s if not parts else s[1:])
        cur = key(s[-1])
        remaining.remove(i)
    W = np.vstack(parts)
    keep = np.concatenate([[True],
                           np.linalg.norm(np.diff(W, axis=0), axis=1) > 1e-12])
    return W[keep]


def _internal_junctions(segs, wall_ends, tol=5e-3):
    """Segment endpoints that are not the wall's own ends = T-junctions."""
    J = []
    for s in segs:
        for p in (np.asarray(s[0], float), np.asarray(s[-1], float)):
            if min(np.linalg.norm(p - e) for e in wall_ends) < tol:
                continue
            if not any(np.linalg.norm(p - q) < 1e-6 for q in J):
                J.append(p)
    return J


def _split_polyline_at(W, cuts):
    """Split polyline W at the given points (assumed on W). Returns segments."""
    marks = []
    for p in cuts:
        seg, t, dist, proj = ps._project_to_polyline(p, W)
        marks.append((seg + t, seg, np.asarray(p, float)))
    marks.sort(key=lambda m: m[0])
    out, cur, ptr = [], [W[0]], 0
    for i in range(len(W) - 1):
        while ptr < len(marks) and marks[ptr][1] == i:
            p = marks[ptr][2]
            if np.linalg.norm(cur[-1] - p) > 1e-12:
                cur.append(p)
            if len(cur) >= 2:
                out.append(np.array(cur))
            cur = [p]
            ptr += 1
        if np.linalg.norm(cur[-1] - W[i + 1]) > 1e-12:
            cur.append(W[i + 1])
    if len(cur) >= 2:
        out.append(np.array(cur))
    return out


def collapse_seam_wedges(mesh, gap=None, wall_tol=0.02):
    """Collapse 3-sided wall wedges (generalized Xiao-style simplification).

    A wedge is a 3-corner triangular region bounded by two separatrices that
    share a common singularity endpoint and a wall segment connecting their
    other endpoints. Dropping the SHORTER separatrix of such a pair turns the
    shared singularity into a wall T-node and the triangle into a quad.

    Generalized beyond the seam-only original: scans BOTH endpoints of every
    separatrix, accepts any domain wall (periodic seam, inlet, outlet, blade)
    as the bounding wall, and groups pairs by the shared singularity endpoint
    (topological) rather than by `singularity_coords`, so wedges formed by
    separatrices of two different singularities that meet at a common corner
    are also caught. Interior connector segments between T-nodes are NOT
    treated as walls. The `gap` parameter is retained for backward
    compatibility but defaults to None (no gap filter)."""
    A, B = _periodic_chains(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    bnd = [np.asarray(s, float) for s in mesh.streamlines[:n_b]]
    seps = [np.asarray(s, float) for s in mesh.streamlines[n_b:]]
    dicts = list(mesh.separatrices)

    sing_pts = [np.asarray(d["singularity_coords"], float)
                for d in dicts if d.get("singularity_coords") is not None]

    def _is_singularity(p, tol=1e-6):
        return any(np.linalg.norm(p - sp) < tol for sp in sing_pts)

    wall_chains = [A, B]
    for b in bnd:
        if b.ndim != 2 or len(b) < 2:
            continue
        if np.all(b[:, 1] < wall_tol) or np.all(b[:, 1] > 1.0 - wall_tol):
            wall_chains.append(b)
    for loop in getattr(mesh, "blade_loops", []) or []:
        loop = np.asarray(loop, float)
        if loop.ndim == 2 and len(loop) >= 2:
            wall_chains.append(loop)

    wall_endpoints = []
    for C in wall_chains:
        is_closed = np.linalg.norm(C[0] - C[-1]) < 1e-9
        wall_endpoints.append((C[0].copy(), C[-1].copy()) if not is_closed
                              else (None, None))

    def _is_open_wall_endpoint(p, ci, tol=1e-6):
        ep0, ep1 = wall_endpoints[ci]
        if ep0 is None:
            return False
        return np.linalg.norm(p - ep0) < tol or np.linalg.norm(p - ep1) < tol

    def _nearest_walls(p):
        out = []
        for ci, C in enumerate(wall_chains):
            seg, t, dist, _ = ps._project_to_polyline(p, C)
            if dist < wall_tol:
                out.append((ci, seg + t, dist))
        return out

    marks = []
    for i, s in enumerate(seps):
        s = np.asarray(s, float)
        for ep, other in ((s[0], s[-1]), (s[-1], s[0])):
            if not _is_singularity(ep):
                continue
            other_is_sing = _is_singularity(other)
            for ci, pos, _dist in _nearest_walls(other):
                # If the docking endpoint is itself a singularity, only accept
                # the dock when it sits at the END of an OPEN wall (e.g. S1 at
                # the L-seam tip). A singularity in the MIDDLE of a wall (e.g.
                # S4 on a closed blade loop) is a multi-arm junction, not a
                # wall-docking wedge tip — collapsing there over-merges and
                # produces pentagonal regions.
                if other_is_sing and not _is_open_wall_endpoint(other, ci):
                    continue
                marks.append((ci, pos, i, np.asarray(other, float),
                              np.asarray(ep, float)))

    bysing = defaultdict(list)
    for ci, pos, i, wall_end, sing in marks:
        bysing[(round(float(sing[0]), 6), round(float(sing[1]), 6))].append(
            (ci, pos, i, wall_end))

    all_pos = defaultdict(list)
    for ci, pos, i, wall_end, sing in marks:
        all_pos[ci].append(pos)

    drop, freed = set(), []
    for (_sx, _sy), lst in bysing.items():
        if len(lst) < 2:
            continue
        for a, b in itertools.combinations(lst, 2):
            ci_a, p_a, i_a, e_a = a
            ci_b, p_b, i_b, e_b = b
            if ci_a != ci_b:
                continue
            lo, hi = (p_a, p_b) if p_a <= p_b else (p_b, p_a)
            if any(lo + 1e-9 < q < hi - 1e-9 for q in all_pos[ci_a]):
                continue
            if gap is not None and np.linalg.norm(e_b - e_a) > gap:
                continue
            j = i_a if _arclen(seps[i_a]) <= _arclen(seps[i_b]) else i_b
            drop.add(j)
            freed.append(e_a if j == i_a else e_b)

    mesh.dropped_wedge_arms = [seps[j].copy() for j in sorted(drop)]
    if not drop:
        return []
    for e in freed:
        hit = [k for k, b in enumerate(bnd)
               if min(np.linalg.norm(b[0] - e), np.linalg.norm(b[-1] - e))
               < 1e-9]
        if len(hit) != 2:
            continue
        k1, k2 = hit
        a, b = bnd[k1], bnd[k2]
        if np.linalg.norm(a[-1] - e) > 1e-9:
            a = a[::-1]
        if np.linalg.norm(b[0] - e) > 1e-9:
            b = b[::-1]
        bnd[k1] = np.vstack([a, b[1:]])
        bnd.pop(k2)
    mesh.streamlines = bnd + [s for k, s in enumerate(seps) if k not in drop]
    mesh.separatrices = [d for k, d in enumerate(dicts) if k not in drop]
    print(f"[wedge] collapsed {len(drop)} wedge arm(s), "
          f"re-joined wall at {len(freed)} junction(s)")
    return mesh.dropped_wedge_arms


def symmetrize_seam_junctions(mesh, tol_cls=0.012, tol_match=0.012,
                              verbose=True):
    """Master-slave seam conformity (DLR sec 2.7). After the snap pass both
    seam walls are split at the T-junctions of the curves that ended there.
    This unions the junction sets mod pitch: the LEFT wall is the master; every
    junction (own or mirrored from the right) is projected onto it, the right
    wall is rebuilt as the EXACT +pitch translate (copy + shift, never
    re-projected), and the endpoints of the docking interior curves are moved
    onto the exact junction coordinates. Mirrored junctions without an interior
    curve stay hanging (T-nodes)."""
    pitch = float(mesh.pitch_norm)
    shift = np.array([pitch, 0.0])
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    bnd = [np.asarray(s, float) for s in mesh.streamlines[:n_b]]
    seps = [np.asarray(s, float) for s in mesh.streamlines[n_b:]]

    A, B = _periodic_chains(mesh)
    side = []
    for poly in bnd:
        dA = max(_dist_to_chain(q, A) for q in poly)
        dB = max(_dist_to_chain(q, B) for q in poly)
        side.append("L" if dA < tol_cls else "R" if dB < tol_cls else None)
    left = [bnd[i] for i in range(n_b) if side[i] == "L"]
    right = [bnd[i] for i in range(n_b) if side[i] == "R"]
    other = [bnd[i] for i in range(n_b) if side[i] is None]
    if not left or not right:
        print("[seam] walls not found -- symmetrization skipped")
        return None

    WL = _chain_segments(left)
    WR = _chain_segments(right)
    conf = max(np.linalg.norm(q - shift - ps._project_to_polyline(
        q - shift, WL)[3]) for q in WR)
    JL = _internal_junctions(left, [WL[0], WL[-1]])
    JR = _internal_junctions(right, [WR[0], WR[-1]])

    # canonical junction set on the master wall
    canon = []                                   # dicts: coord, srcL, srcR
    for p in JL:
        proj = ps._project_to_polyline(p, WL)[3]
        canon.append({"coord": np.asarray(proj, float), "srcL": p, "srcR": None})
    for p in JR:
        q = p - shift
        hit = next((c for c in canon
                    if np.linalg.norm(c["coord"] - q) < tol_match), None)
        if hit is None:
            proj = ps._project_to_polyline(q, WL)[3]
            canon.append({"coord": np.asarray(proj, float),
                          "srcL": None, "srcR": p})
        else:
            hit["srcR"] = p

    left_segs = _split_polyline_at(WL, [c["coord"] for c in canon])
    right_segs = [seg + shift for seg in left_segs]

    # re-terminate interior curves exactly on the (possibly moved) junctions
    moved, max_move = 0, 0.0
    for c in canon:
        for sd, old in (("L", c["srcL"]), ("R", c["srcR"])):
            if old is None:
                continue
            new = c["coord"] if sd == "L" else c["coord"] + shift
            d = float(np.linalg.norm(new - old))
            for k, s in enumerate(seps):
                if np.linalg.norm(s[-1] - old) < 1e-6:
                    seps[k] = np.vstack([s[:-1], new])
                elif np.linalg.norm(s[0] - old) < 1e-6:
                    seps[k] = np.vstack([new, s[1:]])
                else:
                    continue
                moved += 1
                max_move = max(max_move, d)

    mesh.streamlines = other + left_segs + right_segs + seps
    n_mirror = sum(1 for c in canon if c["srcL"] is None or c["srcR"] is None)
    if verbose:
        print(f"[seam] junctions: L={len(JL)} R={len(JR)} union={len(canon)} "
              f"({n_mirror} hanging mirrors); wall conformity pre={conf:.2e} "
              f"post=0 (copy+shift); re-terminated {moved} curve ends "
              f"(max move {max_move:.2e})")
    return {"n_left": len(JL), "n_right": len(JR), "n_union": len(canon),
            "n_hanging": n_mirror, "wall_conformity_pre": conf,
            "max_endpoint_move": max_move, "WL": WL, "WR": WR,
            "canon": canon, "pitch": pitch}


# --------------------------------------------------------------------------
# variant T-b: continue block edges across the seam from hanging junctions
# --------------------------------------------------------------------------

def continue_hanging_junctions(sl, seam_info, min_len=0.02):
    """Emit a streamline into the domain from every hanging mirrored junction:
    the periodic continuation of the curve docking on the opposite seam (same
    direction vector, shifted by one pitch). The emitted curve is snapped like
    a separatrix (point target first, else boundary T-split). Returns the
    number of curves emitted (0 = nothing hanging -> caller stops iterating)."""
    import torch
    mesh = sl.mesh
    pitch = seam_info["pitch"]
    shift = np.array([pitch, 0.0])
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    seps = [np.asarray(s, float) for s in mesh.streamlines[n_b:]]

    emitted = 0
    for c in seam_info["canon"]:
        if c["srcL"] is not None and c["srcR"] is not None:
            continue
        hang_on = "L" if c["srcL"] is None else "R"
        p = c["coord"] if hang_on == "L" else c["coord"] + shift
        src_end = c["coord"] + shift if hang_on == "L" else c["coord"]
        # tangent of the docking curve at the opposite seam (into the wall)
        v = None
        for s in seps:
            if np.linalg.norm(s[-1] - src_end) < 1e-6 and len(s) >= 2:
                v = s[-1] - s[max(0, len(s) - 4)]
                break
            if np.linalg.norm(s[0] - src_end) < 1e-6 and len(s) >= 2:
                v = s[0] - s[min(len(s) - 1, 3)]
                break
        if v is None:
            continue
        n = np.linalg.norm(v)
        if n < 1e-12:
            continue
        v = v / n
        eps = 0.012
        p0 = torch.tensor(p + eps * v, dtype=torch.float)
        dir0 = torch.tensor(v, dtype=torch.float)
        fi = sl.find_containing_face(p0, mesh)
        if fi is None:
            continue
        vec, _ = sl.get_best_cross_vector(p0, dir0, mesh, fi)
        if vec is None:
            continue
        curve = [p.copy(), (p + eps * v).copy()]
        curve = sl.runge_kutta_heun_integrate_streamline(p0, vec, mesh, curve)
        curve = np.asarray(curve, float)
        if len(curve) < 3 or _arclen(curve) < min_len:
            continue
        curve = _dock_continuation(mesh, curve)
        if curve is None:
            continue
        mesh.streamlines.append(curve)
        mesh.separatrices.append({
            "coordinates": torch.tensor(p + eps * v, dtype=torch.float),
            "vector": dir0,
            "singularity_coords": torch.tensor(p, dtype=torch.float),
            "face_id": -8000 - emitted,      # seam-continuation marker
        })
        emitted += 1
    if emitted:
        print(f"[continue] emitted {emitted} seam-continuation curve(s)")
    return emitted


def _seg_intersect(p1, p2, q1, q2, eps=1e-12):
    """Intersection of segments p1p2 and q1q2. Returns (s, t, point) with
    s,t in [0,1] or None."""
    r = p2 - p1
    d = q2 - q1
    den = r[0] * d[1] - r[1] * d[0]
    if abs(den) < eps:
        return None
    dp = q1 - p1
    s = (dp[0] * d[1] - dp[1] * d[0]) / den
    t = (dp[0] * r[1] - dp[1] * r[0]) / den
    if -1e-9 <= s <= 1 + 1e-9 and -1e-9 <= t <= 1 + 1e-9:
        return s, t, p1 + s * r
    return None


def _first_crossing(curve, others, skip_start_len=0.03):
    """First transversal crossing of `curve` with any polyline in `others`.
    Returns (curve_cut_index, cross_point, other_index, other_seg_pos) or
    None. Crossings within skip_start_len of the curve start are ignored (the
    start sits ON the seam wall)."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    best = None
    for oi, poly in enumerate(others):
        poly = np.asarray(poly, float)
        if poly.ndim != 2 or len(poly) < 2:
            continue
        for i in range(len(curve) - 1):
            if cum[i] < skip_start_len:
                continue
            if best is not None and cum[i] > best[0]:
                break
            for j in range(len(poly) - 1):
                hit = _seg_intersect(curve[i], curve[i + 1],
                                     poly[j], poly[j + 1])
                if hit is None:
                    continue
                s, t, pt = hit
                pos = cum[i] + s * seg[i]
                if pos < skip_start_len:
                    continue
                if best is None or pos < best[0]:
                    best = (pos, i, pt, oi, j + t)
    if best is None:
        return None
    _pos, i, pt, oi, segpos = best
    return i, pt, oi, segpos


def _dock_continuation(sl_mesh, curve, radius=0.045, bnd_radius=0.05,
                       dup_tol=0.03):
    """Terminate a freshly emitted seam continuation T-mesh-style:
      1. truncate on the first point target (singularity / c0 corner),
      2. else truncate at the FIRST transversal crossing with any existing
         curve and split that curve there (T-junction; running on to the far
         boundary would slice the whole passage into off-family pieces),
      3. else (no crossing) project the end onto the nearest boundary
         polyline and split it.
    Near-duplicates of an existing separatrix (Hausdorff < dup_tol after
    truncation) are discarded -- the junction stays hanging."""
    mesh = sl_mesh
    nodes = ps._termination_nodes(mesh)
    origin = curve[0]
    for j in range(3, len(curve)):
        if len(nodes) == 0:
            break
        d = np.linalg.norm(nodes - curve[j], axis=1)
        k = int(np.argmin(d))
        if d[k] < radius and np.linalg.norm(nodes[k] - origin) > radius:
            curve = np.vstack([curve[:j], nodes[k]])
            break
    else:
        n_b = len(mesh.streamlines) - len(mesh.separatrices)
        others = [np.asarray(s, float) for s in mesh.streamlines]
        hit = _first_crossing(curve, others)
        if hit is not None:
            i, pt, oi, segpos = hit
            curve = np.vstack([curve[:i + 1], pt])
            poly = others[oi]
            if min(np.linalg.norm(pt - poly[0]),
                   np.linalg.norm(pt - poly[-1])) > 1e-9:
                pieces = _split_polyline_at(poly, [pt])
                if oi < n_b:                     # boundary: plain split
                    mesh.streamlines[oi] = pieces[0]
                    for extra in pieces[1:]:
                        mesh.streamlines.insert(oi + 1, extra)
                else:                            # separatrix: split + clone dict
                    mesh.streamlines[oi] = pieces[0]
                    si = oi - n_b
                    base = mesh.separatrices[si]
                    for pi, extra in enumerate(pieces[1:]):
                        mesh.streamlines.append(extra)
                        mesh.separatrices.append({
                            "coordinates": base.get("coordinates"),
                            "vector": base.get("vector"),
                            "singularity_coords":
                                base.get("singularity_coords"),
                            "face_id": -8500 - si,
                        })
        else:
            end = curve[-1]
            best = (None, np.inf, None)
            for bi in range(n_b):
                poly = np.asarray(mesh.streamlines[bi], float)
                seg, t, dist, proj = ps._project_to_polyline(end, poly)
                if dist < best[1]:
                    best = (bi, dist, proj)
            if best[0] is None or best[1] > bnd_radius:
                return None                      # dangling -> discard
            bi, _, proj = best
            poly = np.asarray(mesh.streamlines[bi], float)
            if min(np.linalg.norm(proj - poly[0]),
                   np.linalg.norm(proj - poly[-1])) > 1e-9:
                pieces = _split_polyline_at(poly, [proj])
                mesh.streamlines[bi] = pieces[0]
                for extra in pieces[1:]:
                    mesh.streamlines.insert(bi + 1, extra)
            curve = np.vstack([curve[:-1], proj])

    if _arclen(curve) < 0.02:
        return None
    # near-duplicate guard (e.g. continuation hugging an existing arm)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    cs_ = curve[:: max(1, len(curve) // 30)]
    for s in mesh.streamlines[n_b:]:
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) < 2:
            continue
        ss = s[:: max(1, len(s) // 30)]
        d = np.linalg.norm(cs_[:, None, :] - ss[None, :, :], axis=2)
        if max(d.min(axis=1).max(), d.min(axis=0).max()) < dup_tol:
            return None
    return curve


# --------------------------------------------------------------------------
# conforming cell counts per graph edge (integer program)
# --------------------------------------------------------------------------

def _edge_key(a, b, e2s):
    return (a, b) if (a, b) in e2s else (b, a)


def solve_edge_divisions(result, seam_pairs, h=H_CELL, min_cells=None):
    """Cells per graph edge, CFD-conforming: opposite sides of every block
    carry the same total cell count, seam edge pairs share their count.
    scipy MILP: minimize sum(c_e), c_e integer >= max(1, round(len_e/h)).

    ``min_cells`` (optional {edge_key: n}) raises the lower bound of specific
    edges -- used to force a boundary-layer cell count on the (short) blade-
    normal edges so the near-wall clustering has enough cells to resolve; the
    conformity constraints then propagate that count to the opposite sides."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    e2s = result["e2s"]
    keys = list(e2s.keys())
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    min_cells = min_cells or {}
    lb = np.array([max(1, int(round(_arclen(e2s[k]) / h)), min_cells.get(k, 1))
                   for k in keys], float)

    rows = []
    for blk in result["blocks"]:
        chains = blk["side_chains"]
        for sa, sb in ((0, 2), (1, 3)):
            row = np.zeros(n)
            for j in range(len(chains[sa]) - 1):
                row[idx[_edge_key(chains[sa][j], chains[sa][j + 1], e2s)]] += 1
            for j in range(len(chains[sb]) - 1):
                row[idx[_edge_key(chains[sb][j], chains[sb][j + 1], e2s)]] -= 1
            if np.any(row):
                rows.append(row)
    for ka, kb in seam_pairs:
        row = np.zeros(n)
        row[idx[ka]] += 1
        row[idx[kb]] -= 1
        rows.append(row)

    A = np.array(rows) if rows else np.zeros((0, n))
    res = milp(c=np.ones(n),
               constraints=[LinearConstraint(A, 0.0, 0.0)] if len(A) else [],
               integrality=np.ones(n),
               bounds=Bounds(lb, np.full(n, np.inf)))
    if not res.success:
        print(f"[milp] WARNING: {res.message} -- falling back to lb (grid "
              f"may be non-conforming)")
        counts = lb
    else:
        counts = np.round(res.x)
    print(f"[milp] edge divisions: {n} edges, "
          f"cells min={int(counts.min())} max={int(counts.max())} "
          f"total-extra={int(counts.sum() - lb.sum())} "
          f"({'optimal' if res.success else 'FALLBACK'})")
    return {k: int(counts[idx[k]]) for k in keys}


# --------------------------------------------------------------------------
# tanh-clustered edge sampling (canonical, one distribution per edge)
# --------------------------------------------------------------------------

def _bisect_beta(first_cell_of_beta, target, lo=1e-3, hi=20.0):
    """Find beta with first_cell_of_beta(beta) ~= target (monotone falling)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if first_cell_of_beta(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def edge_fractions(nc, cluster_start, cluster_end, ratio=CLUSTER_RATIO):
    """Arclength fractions for nc cells (nc+1 points), tanh-clustered towards
    the flagged end(s) so the first wall cell is ~uniform/ratio."""
    x = np.linspace(0.0, 1.0, nc + 1)
    if nc < 2 or (not cluster_start and not cluster_end):
        return x
    target = (1.0 / nc) / ratio

    def one_sided(b):
        """f(x) = 1 + tanh(b(x-1))/tanh(b): clusters at x=0."""
        return 1.0 + np.tanh(b * (x - 1.0)) / np.tanh(b)

    def two_sided(b):
        """f(x) = (1 + tanh(b(2x-1))/tanh(b))/2: clusters at both ends."""
        return 0.5 * (1.0 + np.tanh(b * (2.0 * x - 1.0)) / np.tanh(b))

    if cluster_start and cluster_end:
        b = _bisect_beta(lambda bb: two_sided(bb)[1], target)
        f = two_sided(b)
    elif cluster_start:
        b = _bisect_beta(lambda bb: one_sided(bb)[1], target)
        f = one_sided(b)
    else:                                        # cluster at the END: mirror
        b = _bisect_beta(lambda bb: one_sided(bb)[1], target)
        f = 1.0 - one_sided(b)[::-1]
    f[0], f[-1] = 0.0, 1.0
    return f


def _morph_ends_to(s, pa, pb, frac=0.3, max_win=0.06):
    """Move a sampled edge's endpoints exactly onto the graph node coordinates
    pa/pb, bridging the (node-merge tolerance) offset smoothly over the first/
    last part of the edge. Keeps the curve watertight at block corners."""
    s = np.asarray(s, float).copy()
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    if L < 1e-15:
        return s
    win = min(frac * L, max_win)
    oa = np.asarray(pa, float) - s[0]
    ob = np.asarray(pb, float) - s[-1]
    wa = np.clip(1.0 - cum / win, 0.0, 1.0) ** 2 if win > 0 else \
        (cum == 0.0).astype(float)
    wb = np.clip(1.0 - (L - cum) / win, 0.0, 1.0) ** 2 if win > 0 else \
        (cum == L).astype(float)
    return s + wa[:, None] * oa[None, :] + wb[:, None] * ob[None, :]


def blade_normal_edges(result, wall_dist_fn, seam_pairs, wall_on_tol=1e-3):
    """Keys of edges that dock on the blade at one end but run AWAY from it
    (not along) -- the wall-normal boundary-layer edges. These are the edges
    whose cell count must be raised so the near-wall tanh clustering resolves.
    Seam-twin right edges are mapped back to their left partner key (the count
    is shared, so raising the left raises the right)."""
    e2s = result["e2s"]
    right_of = {kb: ka for ka, kb in seam_pairs}
    out = set()
    for k, poly in e2s.items():
        poly = np.asarray(poly, float)
        d_a = wall_dist_fn(poly[0])
        d_b = wall_dist_fn(poly[-1])
        d_m = wall_dist_fn(poly[len(poly) // 2])
        along = d_m < wall_on_tol
        if not along and (d_a < wall_on_tol or d_b < wall_on_tol):
            out.add(right_of.get(k, k))
    return out


def build_edge_samples(result, divisions, wall_dist_fn, seam_pairs,
                       pitch, wall_on_tol=1e-3, ratio=CLUSTER_RATIO):
    """One canonical sample array per graph edge. Distribution: tanh clustered
    towards endpoints on the CFD wall measured by ``wall_dist_fn`` (the blade
    boundary layer); edges running ALONG the wall stay uniform; edges not
    touching the wall (e.g. inlet/outlet at t=0/1) stay uniform; right-seam
    edges are the exact +pitch copies of their left partners. Endpoints are
    morphed onto the graph node coordinates (watertight corners despite the
    1e-2 node-merge tolerance)."""
    e2s = result["e2s"]
    nodes = result["nodes"]
    samples = {}
    right_of = {kb: ka for ka, kb in seam_pairs}
    for k, poly in e2s.items():
        if k in right_of:
            continue                              # filled from the left twin
        nc = divisions[k]
        poly = np.asarray(poly, float)
        d_a = wall_dist_fn(poly[0])
        d_b = wall_dist_fn(poly[-1])
        d_m = wall_dist_fn(poly[len(poly) // 2])
        along = d_m < wall_on_tol                 # edge lies ON the wall
        cl_a = (d_a < wall_on_tol) and not along
        cl_b = (d_b < wall_on_tol) and not along
        s = _resample_at(poly, edge_fractions(nc, cl_a, cl_b, ratio=ratio))
        samples[k] = _morph_ends_to(s, nodes[k[0]], nodes[k[1]])
    shift = np.array([pitch, 0.0])
    for ka, kb in seam_pairs:
        src = samples[ka]
        polyb = np.asarray(e2s[kb], float)
        cand = src + shift
        if np.linalg.norm(cand[0] - polyb[0]) > np.linalg.norm(
                cand[-1] - polyb[0]):
            cand = cand[::-1]
        samples[kb] = cand
    return samples


def find_seam_edge_pairs(result, WL, WR, pitch, tol=0.01):
    """(left_key, right_key) pairs of wall-segment edges matched mod pitch."""
    e2s = result["e2s"]
    shift = np.array([pitch, 0.0])
    lefts, rights = [], []
    for k, poly in e2s.items():
        poly = np.asarray(poly, float)
        sub = poly[:: max(1, len(poly) // 8)]
        if max(_dist_to_chain(q, WL) for q in sub) < tol:
            lefts.append(k)
        elif max(_dist_to_chain(q, WR) for q in sub) < tol:
            rights.append(k)
    pairs = []
    for ka in lefts:
        ma = np.asarray(e2s[ka], float).mean(axis=0) + shift
        best, bk = np.inf, None
        for kb in rights:
            mb = np.asarray(e2s[kb], float).mean(axis=0)
            d = np.linalg.norm(ma - mb)
            if d < best:
                best, bk = d, kb
        if bk is not None and best < 0.02:
            pairs.append((ka, bk))
    return pairs


# --------------------------------------------------------------------------
# TFI (Coons) with canonical per-edge sampling
# --------------------------------------------------------------------------

def _coons(S, N, W, E):
    """S: c0->c1, N: c3->c2 (both len n_u); W: c0->c3, E: c1->c2 (len n_v).
    Returns grid (n_v, n_u, 2)."""
    n_u, n_v = len(S), len(W)
    u = np.linspace(0.0, 1.0, n_u)[None, :, None]
    v = np.linspace(0.0, 1.0, n_v)[:, None, None]
    c00, c10, c01, c11 = S[0], S[-1], N[0], N[-1]
    X = ((1 - v) * S[None, :, :] + v * N[None, :, :]
         + (1 - u) * W[:, None, :] + u * E[:, None, :]
         - ((1 - u) * (1 - v) * c00 + u * (1 - v) * c10
            + (1 - u) * v * c01 + u * v * c11))
    return X


def _inverted_cells(X):
    """Count TFI cells with non-positive signed area (shoelace)."""
    p00 = X[:-1, :-1]; p10 = X[:-1, 1:]; p11 = X[1:, 1:]; p01 = X[1:, :-1]
    area = 0.5 * ((p00[..., 0] * p10[..., 1] - p10[..., 0] * p00[..., 1])
                  + (p10[..., 0] * p11[..., 1] - p11[..., 0] * p10[..., 1])
                  + (p11[..., 0] * p01[..., 1] - p01[..., 0] * p11[..., 1])
                  + (p01[..., 0] * p00[..., 1] - p00[..., 0] * p01[..., 1]))
    return int((area <= 0).sum()), int(area.size)


def _tm_smooth(X, iters=400, omega=1.0):
    """Thomas-Middlecoff elliptic smoothing of a structured block grid, with
    the four boundary rows/cols held FIXED (so shared block edges and the
    periodic seam stay conforming). Removes the folds a pure Coons patch
    produces under strong wall clustering while the control functions -- taken
    from the boundary point spacing -- carry that near-wall clustering into the
    interior (unlike plain Winslow, which would relax it towards uniform).

    X: (n_eta, n_xi, 2). eta = axis 0, xi = axis 1. Returns the smoothed grid.
    """
    X = np.asarray(X, float).copy()
    n_e, n_x = X.shape[:2]
    if n_e < 3 or n_x < 3:
        return X                                   # no interior to move
    eps = 1e-30

    def _ctrl(a, b):
        """TM source term -(t.t')/(t.t) along a line: a,b are the first/second
        derivative arrays (…,2)."""
        return -(a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) / \
            (a[..., 0] ** 2 + a[..., 1] ** 2 + eps)

    # phi: xi-spacing control from the two eta=const boundaries (rows 0, -1),
    # linearly interpolated across eta.
    phi_bt = []
    for i in (0, -1):
        row = X[i]
        d1 = np.gradient(row, axis=0)              # d/dxi (unit spacing)
        d2 = np.zeros_like(row)
        d2[1:-1] = row[2:] - 2 * row[1:-1] + row[:-2]
        phi_bt.append(_ctrl(d1, d2))
    t = np.linspace(0.0, 1.0, n_e)[:, None]
    phi = (1 - t) * phi_bt[0][None, :] + t * phi_bt[1][None, :]

    # psi: eta-spacing control from the two xi=const boundaries (cols 0, -1),
    # linearly interpolated across xi.
    psi_lr = []
    for j in (0, -1):
        col = X[:, j]
        d1 = np.gradient(col, axis=0)              # d/deta
        d2 = np.zeros_like(col)
        d2[1:-1] = col[2:] - 2 * col[1:-1] + col[:-2]
        psi_lr.append(_ctrl(d1, d2))
    s = np.linspace(0.0, 1.0, n_x)[None, :]
    psi = (1 - s) * psi_lr[0][:, None] + s * psi_lr[1][:, None]

    for _ in range(iters):
        I = X[1:-1, 1:-1]
        xN, xS = X[2:, 1:-1], X[:-2, 1:-1]         # eta +/- (axis 0)
        xE, xW = X[1:-1, 2:], X[1:-1, :-2]          # xi  +/- (axis 1)
        xNE, xNW = X[2:, 2:], X[2:, :-2]
        xSE, xSW = X[:-2, 2:], X[:-2, :-2]
        x_xi = 0.5 * (xE - xW)
        x_eta = 0.5 * (xN - xS)
        alpha = (x_eta ** 2).sum(-1, keepdims=True)
        gamma = (x_xi ** 2).sum(-1, keepdims=True)
        beta = (x_xi * x_eta).sum(-1, keepdims=True)
        x_xieta = 0.25 * (xNE - xNW - xSE + xSW)
        ph = phi[1:-1, 1:-1, None]
        ps = psi[1:-1, 1:-1, None]
        rhs = (alpha * (xE + xW + ph * x_xi)
               + gamma * (xN + xS + ps * x_eta)
               - 2.0 * beta * x_xieta)
        new = rhs / (2.0 * (alpha + gamma) + eps)
        X[1:-1, 1:-1] = (1 - omega) * I + omega * new
    return X


def tfi_fill(result, edge_samples, WL, WR, pitch, seam_tol=0.01, smooth=True):
    """Coons patch per block from the canonical per-edge samples, then optional
    Thomas-Middlecoff elliptic smoothing (``smooth``) to untangle the strong
    blade boundary-layer clustering. Cell counts solved conforming
    (solve_edge_divisions), so a block side is ALWAYS the concatenation of its
    edge samples -- grid points are identical across every block edge and
    across the periodic seam (smoothing keeps block boundaries fixed)."""
    e2s = result["e2s"]

    def _edge_s(a, b):
        k = _edge_key(a, b, e2s)
        s = edge_samples[k]           # oriented like e2s[k]
        return s if k == (a, b) else s[::-1]

    def _side_samples(chain):
        parts = [_edge_s(chain[j], chain[j + 1])
                 for j in range(len(chain) - 1)]
        out = [parts[0]]
        out.extend(p[1:] for p in parts[1:])
        return np.vstack(out)

    def _seam_flag(poly):
        sub = poly[:: max(1, len(poly) // 8)]
        if max(_dist_to_chain(q, WL) for q in sub) < seam_tol:
            return "L"
        if max(_dist_to_chain(q, WR) for q in sub) < seam_tol:
            return "R"
        return None

    grids, n_inv, n_cells = [], 0, 0
    seam_pts = {"L": [], "R": []}
    mismatched = 0
    for blk in result["blocks"]:
        samp = [_side_samples(ch) for ch in blk["side_chains"]]
        if len(samp[0]) != len(samp[2]) or len(samp[1]) != len(samp[3]):
            mismatched += 1
            continue
        S = samp[0]
        E = samp[1]
        N = samp[2][::-1]                 # side2: c2->c3, Coons wants c3->c2
        W = samp[3][::-1]                 # side3: c3->c0, Coons wants c0->c3
        X = _coons(S, N, W, E)
        if smooth:
            X = _tm_smooth(X)
        inv, tot = _inverted_cells(X)
        n_inv += inv
        n_cells += tot
        grids.append(X)
        for i in range(4):
            f = _seam_flag(blk["sides"][i])
            if f:
                seam_pts[f].append(samp[i])
    if mismatched:
        print(f"[tfi] WARNING: {mismatched} block(s) with mismatched side "
              f"counts skipped (MILP fallback?)")

    seam_dev, n_l, n_r = None, 0, 0
    if seam_pts["L"] and seam_pts["R"]:
        Lp = np.unique(np.round(np.vstack(seam_pts["L"]), 12), axis=0)
        Rp = np.unique(np.round(np.vstack(seam_pts["R"]), 12), axis=0) \
            - np.array([pitch, 0.0])
        n_l, n_r = len(Lp), len(Rp)
        d = np.linalg.norm(Lp[:, None, :] - Rp[None, :, :], axis=2)
        seam_dev = float(max(d.min(axis=1).max(), d.min(axis=0).max()))
    return {"grids": grids, "inverted_cells": n_inv, "total_cells": n_cells,
            "seam_dev": seam_dev, "seam_nodes_lr": (n_l, n_r),
            "seam_pts": seam_pts, "mismatched_blocks": mismatched}


def check_edge_conformity(result, edge_samples):
    """Grid conformity across block edges. Both adjacent blocks reference the
    SAME canonical sample array per edge (structural conformity); what can
    still break watertightness are the block CORNERS: build_connectivity
    merges curve endpoints within tol 1e-2 into one node, so two edges meeting
    at a node may end at slightly different coordinates. Returns
    (n_shared_edges, max corner gap = max |edge sample end - node coord|)."""
    e2s = result["e2s"]
    nodes = result["nodes"]
    edge_owner = defaultdict(set)
    for bi, blk in enumerate(result["blocks"]):
        for ch in blk["side_chains"]:
            for j in range(len(ch) - 1):
                edge_owner[_edge_key(ch[j], ch[j + 1], e2s)].add(bi)
    n_shared = sum(1 for o in edge_owner.values() if len(o) >= 2)
    max_gap = 0.0
    for (a, b) in edge_owner:
        s = edge_samples[(a, b)]
        max_gap = max(max_gap,
                      float(np.linalg.norm(s[0] - nodes[a])),
                      float(np.linalg.norm(s[-1] - nodes[b])))
    return n_shared, max_gap


def wall_cell_ratio(tfi_grids_or_result, edge_samples, result, nonper_dist_fn,
                    tol=1e-3):
    """Mean/target check of the boundary-layer clustering: for edges STARTING
    or ENDING on a non-periodic wall, ratio = uniform spacing / first cell."""
    ratios = []
    for k, s in edge_samples.items():
        nc = len(s) - 1
        if nc < 3:
            continue
        L = _arclen(s)
        uni = L / nc
        if nonper_dist_fn(s[0]) < tol and nonper_dist_fn(s[len(s) // 2]) >= tol:
            ratios.append(uni / max(np.linalg.norm(s[1] - s[0]), 1e-30))
        if nonper_dist_fn(s[-1]) < tol and nonper_dist_fn(s[len(s) // 2]) >= tol:
            ratios.append(uni / max(np.linalg.norm(s[-1] - s[-2]), 1e-30))
    return (float(np.mean(ratios)), len(ratios)) if ratios else (None, 0)


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def run_tmesh(stl, out_dir, verbose=True,
              continue_seam_edges=False, max_rounds=3, tag="ta",
              prescribed_singularities=None, flat_tol_deg=15.0):
    ps.set_periodic(True)          # field seam weld stays ON
    ps.set_tile_periodic(False)    # block stage: seam = wall
    t0 = time.time()

    mesh, transform = build_dp_data(stl)
    pitch = float(mesh.pitch_norm)
    ff = FrameField(mesh)
    m = detect_singularities(ff.mesh)

    if prescribed_singularities:
        from . import field_solver as fs
        ff.mesh = fs.enforce_singularities(ff.mesh, prescribed_singularities, transform)
        n_sing = int((ff.mesh.singularities != 0).sum())
    else:
        n_sing = int((m.singularities != 0).sum())

    sl = StreamlineGenerator_v2(ff.mesh)
    ps._drop_degenerate_corner_seps(sl.mesh)
    ps._snap_separatrix_endpoints(sl.mesh, radius=0.045)
    fix_start_kinks(sl.mesh)
    straighten_sing_connectors(sl.mesh)

    collapse_seam_wedges(sl.mesh)
    seam_info = symmetrize_seam_junctions(sl.mesh)

    if continue_seam_edges and seam_info:
        for rnd in range(max_rounds):
            emitted = continue_hanging_junctions(sl, seam_info)
            if not emitted:
                break
            seam_info = symmetrize_seam_junctions(sl.mesh, verbose=False)
        n_still = seam_info["n_hanging"]
        print(f"[continue] after {rnd + 1} round(s): "
              f"{n_still} hanging junction(s) remain")

    n_boundary = len(sl.mesh.streamlines) - len(sl.mesh.separatrices)
    boundary_ref = [np.asarray(s, float)
                    for s in sl.mesh.streamlines[:n_boundary]]
    WL, WR = (seam_info["WL"], seam_info["WR"]) if seam_info else (None, None)

    def _bdist(p):
        return ps._min_boundary_dist(p, boundary_ref)

    def _nonper_dist(p):
        """Distance to the NON-periodic boundaries (blade, inlet, outlet)."""
        p = np.asarray(p, float)
        d = min(abs(p[1] - 0.0), abs(p[1] - 1.0))       # inlet/outlet t=0/1
        for bl in mesh.blade_loops:
            d = min(d, tmf._dist_to_polyline(p, np.asarray(bl, float)))
        return d

    def _blade_dist(p):
        """Distance to the BLADE only (CFD wall). Inlet/outlet at t=0/1 are NOT
        included, so only edges reaching the blade get boundary-layer
        clustering; inlet/outlet edges stay uniform."""
        p = np.asarray(p, float)
        d = np.inf
        for bl in mesh.blade_loops:
            d = min(d, tmf._dist_to_polyline(p, np.asarray(bl, float)))
        return d

    resample_coarse_separatrices(sl.mesh)
    merging = StreamlineMerging(sl.mesh, verbose=bool(os.environ.get("DP3D_DEBUG")))
    splitter = StreamlineIntersectionSplitter(offset_boundingBox=0.05,
                                              num_samples=5)
    updated = splitter.process_streamlines(merging.new_streamlines)

    gen = tmf.TMeshFaceGenerator(updated, blade_loops=list(mesh.blade_loops),
                                 flat_tol_deg=flat_tol_deg, verbose=verbose,
                                 boundary_dist_fn=_bdist, bnd_tol=1e-3)
    result = gen.get_blocks()
    for rej in result["rejects"]:
        nds = result["nodes"][rej["cycle"]] if rej.get("cycle") else []
        turns = (tmf.corner_turns(rej["cycle"], result["e2s"])
                 if rej.get("cycle") and rej.get("ring") is not None else [])
        print(f"[tmesh] reject n_real={rej['n_real']} cycle={rej['cycle']} "
              f"nodes={np.round(np.asarray(nds), 3).tolist()} "
              f"turns={np.round(turns, 1).tolist()}")

    regular, tnodes, irregular = tmf.node_regularity(result, _bdist)

    tfi, divisions, edge_samples, seam_pairs = None, None, None, []
    n_shared, edge_dev, ratio, n_walledges = 0, None, None, 0
    if seam_info and result["blocks"]:
        seam_pairs = find_seam_edge_pairs(result, WL, WR, pitch)
        bl_edges = blade_normal_edges(result, _blade_dist, seam_pairs)
        min_cells = {k: BLADE_NORMAL_CELLS for k in bl_edges}
        divisions = solve_edge_divisions(result, seam_pairs,
                                         min_cells=min_cells)
        edge_samples = build_edge_samples(result, divisions, _blade_dist,
                                          seam_pairs, pitch,
                                          ratio=BLADE_CLUSTER_RATIO)
        tfi = tfi_fill(result, edge_samples, WL, WR, pitch)
        n_shared, edge_dev = check_edge_conformity(result, edge_samples)
        ratio, n_walledges = wall_cell_ratio(tfi, edge_samples, result,
                                             _blade_dist)

    # block-corner seam conformity (block corners on the seams, mod pitch)
    corner_dev, corner_lr = None, (0, 0)
    if seam_info and result["blocks"]:
        cn = np.unique(np.concatenate([b["corners"] for b in result["blocks"]]))
        pts = result["nodes"][cn]
        Lc = np.array([p for p in pts if _dist_to_chain(p, WL) < 0.01])
        Rc = np.array([p for p in pts if _dist_to_chain(p, WR) < 0.01])
        corner_lr = (len(Lc), len(Rc))
        if len(Lc) and len(Rc):
            d = np.linalg.norm(Lc[:, None, :]
                               - (Rc[None, :, :] - np.array([pitch, 0.0])),
                               axis=2)
            corner_dev = float(max(d.min(axis=1).max(), d.min(axis=0).max()))

    runtime = time.time() - t0
    metrics = {
        "approach": ("T-b (seam edges continued)" if continue_seam_edges
                     else "T-a (hanging seam T-nodes)"),
        "singularities": n_sing,
        "blocks": len(result["blocks"]),
        "rejected_regions": len(result["rejects"]),
        "t_nodes_interior": len(tnodes),
        "hanging_seam_junctions": seam_info["n_hanging"] if seam_info else None,
        "irregular_interior_nodes": len(irregular),
        "inverted_blocks": sum(1 for b in result["blocks"] if b["area"] <= 0),
        "inverted_tfi_cells": tfi["inverted_cells"] if tfi else None,
        "total_tfi_cells": tfi["total_cells"] if tfi else None,
        "shared_edges": n_shared,
        "edge_conformity_dev": edge_dev,
        "wall_cluster_ratio_mean": ratio,
        "wall_clustered_edge_ends": n_walledges,
        "seam_corner_lr": list(corner_lr),
        "seam_corner_dev": corner_dev,
        "seam_tfi_nodes_lr": list(tfi["seam_nodes_lr"]) if tfi else None,
        "seam_tfi_dev": tfi["seam_dev"] if tfi else None,
        "planar": result["planar"],
        "runtime_s": round(runtime, 1),
    }
    if verbose:
        print(f"[tmesh:{tag}] GATE regions!=4 real corners: "
              f"{metrics['rejected_regions']} "
              f"({'PASS' if metrics['rejected_regions'] == 0 else 'FAIL'})")
        print(f"[tmesh:{tag}] GATE edge conformity: {n_shared} shared edges, "
              f"dev={edge_dev}  seam TFI dev={metrics['seam_tfi_dev']}")
        print(f"[tmesh:{tag}] GATE inverted TFI cells: "
              f"{metrics['inverted_tfi_cells']}/{metrics['total_tfi_cells']}")
        print(f"[tmesh:{tag}] wall clustering ratio ~"
              f"{ratio if ratio is None else round(ratio, 2)} "
              f"on {n_walledges} wall edge ends")
        print(f"[tmesh:{tag}] {metrics['blocks']} blocks, "
              f"{metrics['hanging_seam_junctions']} hanging seam junctions, "
              f"{len(irregular)} irregular interior, {runtime:.0f}s")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"tmesh_metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}/tmesh_metrics_{tag}.json")

    return {"metrics": metrics, "result": result, "tfi": tfi, "mesh": mesh,
            "seam_info": seam_info, "boundary_ref": boundary_ref,
            "edge_samples": edge_samples, "divisions": divisions,
            "transform": transform, "seam_pairs": seam_pairs,
            "merging": merging}
