#!/usr/bin/env python3
"""
Stage 1 / Step A: Unwrap a cylindrical turbine surface (hub or shroud) into a
flat 2D domain.

Hub and shroud of T1_9 are exact cylinders (constant radius). The mapping

    theta = atan2(y, x)
    s     = r * theta        (circumferential arc length)
    t     = z                (axial)

is isometric (distortion-free), so the 2D triangulation is identical to the 3D
one - only the vertex coordinates change. The result is a rectangular-ish domain
with a blade-shaped hole, i.e. exactly the domain_partition 2D case.

Outputs (returned by ``unwrap``):
    points3d   (N,3)  welded 3D coordinates
    st         (N,2)  unwrapped (s, t) coordinates
    tris       (M,3)  triangle connectivity (0-based, into welded points)
    r          float  cylinder radius
    loops      list[list[int]]  ordered boundary node loops (outer first)
    node_dim   (N,)   2 = interior, 1 = boundary, 0 = corner
    corner_type (N,)  -1 = not a corner, 0 = outer-domain corner,
                      1 = blade-tip LE/TE corner
    blade_loops list[(n,2)]  inner blade loop outlines in (s,t)

Run directly to dump a diagnostic PNG/JSON for the hub.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import meshio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# When False, the inner blade loop(s) are NOT given LE/TE corner nodes
# (corner_type==1). The genuine field singularities the cross field places at
# each tip then drive the partition instead of an artificial tip-corner. The
# blade boundary stays a single smooth closed streamline (separatrices snapping
# onto it create the needed T-junctions). Set via set_blade_tip_corners().
MARK_BLADE_TIP_CORNERS = True


def set_blade_tip_corners(flag):
    global MARK_BLADE_TIP_CORNERS
    MARK_BLADE_TIP_CORNERS = bool(flag)


def _weld(points, tris, decimals=6):
    """Merge coincident STL vertices, remap triangles."""
    key = np.round(points, decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, inv[tris]


def _ensure_ccw_winding(pts, tris):
    """Detect clockwise winding and flip triangles to counter-clockwise.

    For a cylindrical surface centered at the origin, CCW vertex ordering
    produces outward-pointing normals (radially away from the z-axis).  If
    the mesh has CW ordering (inward normals), each triangle [a,b,c] is
    rewritten as [a,c,b] so that the normal flips outward.
    """
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]

    # Normal from cross product of edges
    normals = np.cross(v1 - v0, v2 - v0)

    # Radial direction at triangle centroid (xy-plane only)
    centroids = (v0 + v1 + v2) / 3.0
    radial = centroids[:, :2]
    r_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    r_norm[r_norm == 0] = 1.0
    radial = radial / r_norm

    # Positive dot => normal points outward => CCW
    dot = np.einsum("ij,ij->i", normals[:, :2], radial)

    n_total = len(tris)
    n_cw = int(np.sum(dot < 0))
    n_ccw = n_total - n_cw

    if n_cw > n_ccw:
        print(f"[unwrap] CW winding detected ({n_cw}/{n_total}), flipping to CCW")
        tris = tris.copy()
        tris[:, [1, 2]] = tris[:, [2, 1]]
        return tris
    else:
        print(f"[unwrap] CCW winding confirmed ({n_ccw}/{n_total})")
        return tris


def _boundary_edges(tris):
    """Edges incident to exactly one triangle."""
    cnt = defaultdict(int)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            cnt[(min(u, v), max(u, v))] += 1
    return [e for e, n in cnt.items() if n == 1]


def _trace_loops(bnd_edges):
    """Order boundary edges into closed loops (assumes manifold deg-2 boundary)."""
    adj = defaultdict(list)
    for a, b in bnd_edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def _polygon_area(st):
    """Signed shoelace area of an ordered (n,2) polygon."""
    x, y = st[:, 0], st[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _chord_extremes(loop, st):
    """Return the loop node-pair that is farthest apart (blade LE/TE)."""
    pts = st[loop]
    # O(n^2) on a ~100-node loop is fine.
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return [loop[i], loop[j]]


def _split_loop_segments(loop, corner_set):
    """Split a cyclic node loop into corner->corner node segments (inclusive)."""
    n = len(loop)
    pos = [i for i, v in enumerate(loop) if v in corner_set]
    segs = []
    for k in range(len(pos)):
        i0, i1 = pos[k], pos[(k + 1) % len(pos)]
        seg, i = [], i0
        while True:
            seg.append(loop[i])
            if i == i1:
                break
            i = (i + 1) % n
        segs.append(seg)
    return segs


def _detect_periodic_pair(st, loop, corners, tol=1e-3):
    """Find the pitchwise-periodic boundary pair on the outer loop.

    The unwrapped passage is a sheared parallelogram: two outer sides are the
    axial walls (constant t) and two are the theta-pitch walls. The pitch walls
    are pure s-translates of each other (partner of (s,t) is (s+pitch, t)). We
    locate the two corner->corner segments spanning the full t-range and, if they
    are conforming translates, return node-id pairs (master_left, slave_right)
    sorted by t plus the pitch. Returns ([], None) if no clean conforming pair
    exists (e.g. non-conforming seam -> would need t-interpolation)."""
    trange = st[:, 1].max() - st[:, 1].min()
    if trange <= 0:
        return [], None
    segs = _split_loop_segments(loop, set(corners))
    cand = [sg for sg in segs
            if (st[sg][:, 1].max() - st[sg][:, 1].min()) > 0.8 * trange]
    if len(cand) != 2:
        return [], None
    A, B = cand
    if len(A) != len(B):
        return [], None  # non-conforming; defer to interpolation variant
    # order master/slave by mean s (master = left / smaller s)
    if st[A][:, 0].mean() > st[B][:, 0].mean():
        A, B = B, A
    oa = np.argsort(st[A][:, 1])
    ob = np.argsort(st[B][:, 1])
    master = [A[i] for i in oa]
    slave = [B[i] for i in ob]
    sm, sl = st[master], st[slave]
    if np.max(np.abs(sm[:, 1] - sl[:, 1])) > tol:
        return [], None  # t does not match -> not a translate pair
    ds = sl[:, 0] - sm[:, 0]
    if np.std(ds) > tol:
        return [], None  # not a pure s-translate
    pitch = float(np.mean(ds))
    pairs = [(int(m), int(s)) for m, s in zip(master, slave)]
    return pairs, pitch


def _detect_corners(loop, st, angle_thresh_deg=40.0):
    """Flag loop vertices whose turning angle exceeds threshold as corners."""
    pts = st[loop]
    n = len(loop)
    corners = []
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        v1 = b - a
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        turn = np.degrees(np.arccos(cosang))
        if turn > angle_thresh_deg:
            corners.append(loop[i])
    return corners


def unwrap(stl_path, corner_angle_deg=40.0):
    mesh = meshio.read(str(stl_path))
    raw_pts = mesh.points
    tris_raw = np.vstack([c.data for c in mesh.cells if c.type == "triangle"])
    pts, tris = _weld(raw_pts, tris_raw)
    tris = _ensure_ccw_winding(pts, tris)

    r_per = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    r = float(np.mean(r_per))
    if r_per.std() > 1e-3 * r:
        print(f"WARNING: radius not constant (std={r_per.std():.4g}); "
              f"surface may not be a true cylinder.")

    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Guard against the +/-pi branch cut (safe for T1_9: theta well inside (-pi,pi)).
    if theta.max() - theta.min() > 1.9 * np.pi:
        raise ValueError("theta spans the branch cut; unwrap needs a cut shift.")
    s = r * theta
    t = pts[:, 2]
    st = np.column_stack([s, t])

    loops = _trace_loops(_boundary_edges(tris))
    # Outer loop = largest absolute enclosed area in (s,t).
    loops.sort(key=lambda lp: abs(_polygon_area(st[lp])), reverse=True)

    node_dim = np.full(len(pts), 2, dtype=np.int64)
    # corner_type: -1 = not a corner, 0 = outer-domain corner (outer loop),
    # 1 = blade-tip LE/TE corner (inner/blade loop). Lets downstream emit
    # separatrices differently per corner kind.
    corner_type = np.full(len(pts), -1, dtype=np.int64)
    loop_corners = []
    for li, lp in enumerate(loops):
        for v in lp:
            node_dim[v] = 1
        # Inner blade loops: optionally leave them smooth (no LE/TE corners) so
        # the field's tip singularities drive the partition instead.
        if li >= 1 and not MARK_BLADE_TIP_CORNERS:
            loop_corners.append([])
            continue
        corners = list(dict.fromkeys(_detect_corners(lp, st, corner_angle_deg)))
        # Every loop must carry >=2 corners so it can be split into segments
        # whose endpoints are termination nodes (StreamlineMerging needs this).
        # A smooth blade hole has no angular corners -> anchor LE/TE instead.
        if len(corners) < 2:
            for v in _chord_extremes(lp, st):
                if v not in corners:
                    corners.append(v)
        loop_corners.append(corners)
        ctype = 0 if li == 0 else 1  # loops sorted outer-first
        for v in corners:
            node_dim[v] = 0
            corner_type[v] = ctype

    # blade_loops: the inner (blade) loop outlines as ordered (s,t) polygons,
    # for "does this direction point into the blade profile?" tests downstream.
    blade_loops = [st[lp] for lp in loops[1:]]

    # pitchwise-periodic boundary pair on the outer loop (theta walls). Empty if
    # no clean conforming translate pair exists.
    periodic_pairs, pitch = _detect_periodic_pair(st, loops[0], loop_corners[0])
    if periodic_pairs:
        print(f"[unwrap] periodic theta-pair: {len(periodic_pairs)} node pairs, "
              f"pitch={pitch:.5f}")
    else:
        print("[unwrap] no conforming periodic theta-pair detected")

    return {
        "points3d": pts,
        "st": st,
        "tris": tris,
        "r": r,
        "loops": loops,
        "loop_corners": loop_corners,
        "node_dim": node_dim,
        "corner_type": corner_type,
        "blade_loops": blade_loops,
        "periodic_pairs": periodic_pairs,
        "pitch": pitch,
    }


def _diagnostic(stl_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = unwrap(stl_path)
    st, loops, node_dim = data["st"], data["loops"], data["node_dim"]
    ctype = data["corner_type"]
    print(f"r={data['r']:.4f}  N={len(st)}  tris={len(data['tris'])}")
    print(f"loops: {[len(l) for l in loops]}  (outer first)")
    print(f"corners: {(node_dim == 0).sum()}  boundary: {(node_dim == 1).sum()}")
    print(f"  outer corners: {(ctype == 0).sum()}  blade-tip corners: {(ctype == 1).sum()}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.triplot(st[:, 0], st[:, 1], data["tris"], lw=0.2, color="0.7")
        colors = ["C0", "C1", "C2", "C3"]
        for i, lp in enumerate(loops):
            ring = st[lp + [lp[0]]]
            ax.plot(ring[:, 0], ring[:, 1], colors[i % 4], lw=1.5,
                    label=f"loop {i} (n={len(lp)})")
        outer = ctype == 0
        tip = ctype == 1
        ax.scatter(st[outer, 0], st[outer, 1], c="red", s=45, zorder=5,
                   label="outer corners")
        ax.scatter(st[tip, 0], st[tip, 1], c="magenta", s=70, marker="*",
                   zorder=6, label="blade-tip (LE/TE)")
        ax.set_aspect("equal")
        ax.set_xlabel("s = r*theta")
        ax.set_ylabel("t = z")
        ax.legend()
        ax.set_title(f"Unwrapped {Path(stl_path).name}")
        png = out_dir / "unwrap_diagnostic.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        print(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    meta = {
        "r": data["r"],
        "n_nodes": int(len(st)),
        "n_tris": int(len(data["tris"])),
        "loop_sizes": [len(l) for l in loops],
        "n_corners": int((node_dim == 0).sum()),
    }
    (out_dir / "unwrap_meta.json").write_text(json.dumps(meta, indent=2))


