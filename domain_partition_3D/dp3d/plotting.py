"""Showcase plots: one consistent presentation-style series (transparent
background, no axes/titles) walking through every pipeline stage.

Steps 1-7 are method-independent (all methods share field and integration up
to the endpoint snap) and are written once per part; steps 8-11 are written
per method from the completed run dicts. Entry point: write_plots().
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from . import partition_surface as ps
from . import tmesh_faces as tmf

def _seam_left_fn(mesh):
    """Left-seam profile s_left(t) from the periodic node pairs (master =
    left seam). The theta periodicity is a pure s-translation, so the domain
    column at height t is [s_left(t), s_left(t) + pitch_norm]. The real seam
    is curved, so this must not be approximated by a straight line."""
    pp = getattr(mesh, "periodic_pairs", None)
    if pp is None or len(pp) == 0:
        return lambda t: np.zeros_like(np.asarray(t, float))
    left = mesh.x[pp[:, 0], 0:2].numpy()
    o = np.argsort(left[:, 1])
    ts, ss = left[o, 1], left[o, 0]
    return lambda t: np.interp(np.asarray(t, float), ts, ss)


def fold_curve(s, pitch, s_left, eps=0.02):
    """Fold a cover-coordinate polyline into the fundamental pitch column,
    splitting it wherever it crosses a seam. The column coordinate
    ``u = s - s_left(t)`` uses the true (curved) seam profile, so every folded
    point lands exactly in [s_left(t), s_left(t) + pitch]. The crossing point
    is interpolated and appended to BOTH adjacent segments, so every winding
    is drawn seam-to-seam with no visual gap.

    Points within ``eps`` of a seam keep the previous point's winding index
    (hysteresis): a curve running ALONG a seam jitters around the fold
    boundary numerically and would otherwise fragment into segments landing
    alternately on the left and right seam."""
    s = np.asarray(s, float)
    u = s[:, 0] - s_left(s[:, 1])
    k = np.floor(u / pitch).astype(int)
    for i in range(1, len(s)):
        if k[i] != k[i - 1]:
            m = u[i] % pitch
            if min(m, pitch - m) < eps:
                k[i] = k[i - 1]
    segs, cur = [], [s[0]]
    for i in range(1, len(s)):
        if k[i] != k[i - 1]:
            ub = pitch * max(k[i], k[i - 1])
            f = (ub - u[i - 1]) / (u[i] - u[i - 1] + 1e-30)
            pc = s[i - 1] + np.clip(f, 0.0, 1.0) * (s[i] - s[i - 1])
            cur.append(pc)
            segs.append((k[i - 1], np.asarray(cur)))
            cur = [pc, s[i]]
        else:
            cur.append(s[i])
    segs.append((k[-1], np.asarray(cur)))
    out = []
    for kk, seg in segs:
        seg = seg.copy()
        seg[:, 0] -= kk * pitch
        if len(seg) >= 2:
            out.append(seg)
    return out


def _fold_streamline_unique(s, pitch, s_left, _ends_ok=None,
                            trim_trailing_partial=False,
                            max_wraps=ps.MAX_WRAPS):
    """Fold a streamline and deduplicate periodic closed-orbit stacked copies.

    For a closed orbit wrapping |k|>=2 pitches, fold_curve emits |k| near-identical
    stacked segments. This helper detects such orbits by endpoint match modulo
    pitch (|Δs − k·pitch| < 0.05 AND |Δt| < 0.05, |k| >= 2) and collapses them
    to one representative winding (the longest full segment).

    |k| == 1 closed orbits are NOT collapsed: they form a single geometric loop
    split by seam crossings, with no stacked copies to remove.

    Open separatrices are preserved. If trim_trailing_partial=True and the
    endpoint is NOT _ends_ok, the trailing partial segment is dropped (preserves
    today's step07 behavior).
    """
    s = np.asarray(s, float)
    segs = fold_curve(s, pitch, s_left)
    if len(segs) <= 1:
        return segs
    p0, pe = s[0], s[-1]
    ds = pe[0] - p0[0]
    dt = pe[1] - p0[1]
    k_match = None
    for k in range(-max_wraps, max_wraps + 1):
        if k == 0:
            continue
        if abs(ds - k * pitch) < 0.05 and abs(dt) < 0.05:
            k_match = k
            break
    if k_match is not None:
        if abs(k_match) >= 2:
            return [max(segs, key=len)]
        # |k| == 1: single geometric loop, no stacked copies — keep all segments
        return segs
    # Open separatrix branch
    if trim_trailing_partial and _ends_ok is not None and segs and not _ends_ok(s[-1]):
        segs = segs[:-1]
    return segs


def _on_seam(s, pitch, s_left, eps=0.02):
    """True when every point of the curve lies within eps of a periodic seam
    (u = 0 mod pitch). Such a curve duplicates the seam boundary and is not
    drawn as a colored streamline."""
    s = np.asarray(s, float)
    u = (s[:, 0] - s_left(s[:, 1])) % pitch
    return bool(np.all(np.minimum(u, pitch - u) < eps))


def _streamline_segs(s, pitch, s_left, _ends_ok=None,
                     trim_trailing_partial=False, eps=0.02):
    """Drawable segments of a streamline, kept inside the fundamental column.

    A curve that essentially lives in one pitch column (s-extent <= 1.5
    pitch) is drawn in its dominant column as one continuous curve: a small
    spill across a seam is clipped at the seam, NOT re-inserted at the
    opposite side. Only genuine multi-wrap curves (spirals) fall back to
    seam folding, where every winding runs seam-to-seam."""
    s = np.asarray(s, float)
    u = s[:, 0] - s_left(s[:, 1])
    if u.max() - u.min() > 1.5 * pitch:
        return _fold_streamline_unique(
            s, pitch, s_left, _ends_ok=_ends_ok,
            trim_trailing_partial=trim_trailing_partial)
    k0 = int(np.floor(np.median(u) / pitch))
    uc = u - k0 * pitch
    pts = s.copy()
    pts[:, 0] = s_left(s[:, 1]) + np.clip(uc, 0.0, pitch)
    ins = (uc > -eps) & (uc < pitch + eps)

    def _cross(i, bound):
        f = (bound - uc[i - 1]) / (uc[i] - uc[i - 1] + 1e-30)
        p = s[i - 1] + np.clip(f, 0.0, 1.0) * (s[i] - s[i - 1])
        p = p.copy()
        p[0] -= k0 * pitch
        return p

    segs, cur = [], []
    for i in range(len(pts)):
        if ins[i]:
            if not cur and i > 0 and not ins[i - 1]:
                cur.append(_cross(i, 0.0 if uc[i - 1] < 0 else pitch))
            cur.append(pts[i])
        elif cur:
            cur.append(_cross(i, 0.0 if uc[i] < 0 else pitch))
            segs.append(np.asarray(cur))
            cur = []
    if cur:
        segs.append(np.asarray(cur))
    return [g for g in segs if len(g) >= 2]


def _dedup_pitch_shifted(streamlines, pitch, max_wraps=ps.MAX_WRAPS,
                          tol=0.12):
    """Deduplicate streamlines that are pitch-shifted copies of each other.

    For each entry A, walks through later entries and, for each k in
    ±1..±max_wraps, translates B by k·pitch in the s-coordinate and tests
    via symmetric Hausdorff distance (max of the two directed Hausdorff
    distances) whether B matches A within tol.  A bounding-box overlap pre-
    check rejects obviously unrelated pairs cheaply.  tol must absorb the
    numerical drift between pitch-shifted integrations (~0.1) yet stay below
    the closest distinct-streamline distance (~0.17).

    Returns a list keeping one representative per equivalence group: the
    earliest-indexed entry.  Stable order preserved.
    """
    from scipy.spatial.distance import directed_hausdorff

    items = list(streamlines)
    n = len(items)
    keep, used = [], [False] * n
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        keep.append(items[i])
        A = np.asarray(items[i], float)
        a_min, a_max = A.min(axis=0), A.max(axis=0)
        for j in range(i + 1, n):
            if used[j]:
                continue
            B = np.asarray(items[j], float)
            b_min, b_max = B.min(axis=0), B.max(axis=0)
            for k in range(-max_wraps, max_wraps + 1):
                if k == 0:
                    continue
                ds = k * pitch
                # bbox overlap (B shifted by ds in s)
                if (a_min[0] - tol > b_max[0] + ds or
                        b_min[0] + ds - tol > a_max[0] or
                        a_min[1] - tol > b_max[1] or
                        b_min[1] - tol > a_max[1]):
                    continue
                B_shift = B.copy()
                B_shift[:, 0] += ds
                d_ab = directed_hausdorff(A[:, :2], B_shift[:, :2])[0]
                d_ba = directed_hausdorff(B_shift[:, :2], A[:, :2])[0]
                if max(d_ab, d_ba) < tol:
                    used[j] = True
                    break
    return keep


def get_singularities(mesh):
    """Unique singularity positions from separatrix metadata, labeled S1..Sn
    by ascending x."""
    sings = {}
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    for i, d in enumerate(mesh.separatrices):
        sc = d.get("singularity_coords")
        if sc is None:
            continue
        key = (round(float(sc[0]), 6), round(float(sc[1]), 6))
        if key not in sings:
            sings[key] = {"coords": np.asarray(sc, float),
                          "sep_indices": [], "sep_targets": []}
        idx = n_b + i
        s = np.asarray(mesh.streamlines[idx], float)
        sings[key]["sep_indices"].append(idx)
        sings[key]["sep_targets"].append((idx, s[-1].copy()))
    items = sorted(sings.items(), key=lambda kv: kv[1]["coords"][0])
    labeled = {}
    for li, (key, data) in enumerate(items):
        name = f"S{li + 1}"
        labeled[name] = data
        labeled[name]["name"] = name
    return labeled


def find_target_singularity(target_point, singularities, tol=0.02):
    for name, data in singularities.items():
        if np.linalg.norm(target_point - data["coords"]) < tol:
            return name
    return None


def singularity_connections(mesh, singularities):
    """{(Sa, Sb): [streamline indices]} of direct singularity-singularity
    separatrix connections."""
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    connections = defaultdict(list)
    for i, d in enumerate(mesh.separatrices):
        idx = n_b + i
        s = np.asarray(mesh.streamlines[idx], float)
        if s.ndim != 2 or len(s) < 2:
            continue
        sc = d.get("singularity_coords")
        if sc is None:
            continue
        src = None
        for name, data in singularities.items():
            if np.linalg.norm(data["coords"] - np.asarray(sc, float)) < 0.01:
                src = name
                break
        tgt = find_target_singularity(s[-1], singularities)
        if src and tgt and src != tgt:
            connections[tuple(sorted([src, tgt]))].append(idx)
    return connections


def _save(fig, out_png):
    fig.savefig(out_png, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _draw_domain(ax, st, tris, loops):
    """Shared background: faint triangulation + black boundary contours."""
    ax.triplot(st[:, 0], st[:, 1], tris, lw=0.1, color="black")
    for loop in loops:
        ring = st[loop + [loop[0]]]
        ax.plot(ring[:, 0], ring[:, 1], lw=2.0, color="black")


# --------------------------------------------------------------------------
# steps 1-7: method-independent
# --------------------------------------------------------------------------

def showcase_3d_surface(stl_path, out_png):
    """step01: 3D input surface."""
    import meshio
    m = meshio.read(str(stl_path))
    points = m.points
    triangles = m.cells[0].data

    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=triangles, color="lightgray", alpha=0.8,
                    edgecolor="none")
    ax.set_axis_off()
    spans = points.max(axis=0) - points.min(axis=0)
    ax.set_box_aspect((spans[0], spans[1], spans[2]))
    _save(fig, out_png)


def showcase_unwrapped(st, tris, loops, out_png):
    """step02: unwrapped (s, t) domain."""
    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_crossfield(mesh, st, tris, loops, out_png):
    """step03: 4-RoSy cross-field on boundary nodes."""
    import torch
    boundary_ids = torch.unique(
        mesh.edge_index[0, mesh.edge_attr == 1]).numpy()
    u = mesh.u.numpy()
    base = np.arctan2(u[:, 1], u[:, 0]) / 4.0
    bxy = st[boundary_ids]

    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    for k in range(2):
        ang = base[boundary_ids] + k * (np.pi / 2)
        ax.quiver(bxy[:, 0], bxy[:, 1], np.cos(ang), np.sin(ang),
                  color="#3aa6d0", scale=18, width=0.004,
                  headwidth=0, headlength=0, pivot="mid", alpha=0.95)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_representatives(mesh, st, tris, loops, out_png):
    """step04: boundary cross-field with the representative vector per node."""
    import torch
    boundary_ids = torch.unique(
        mesh.edge_index[0, mesh.edge_attr == 1]).numpy()
    u = mesh.u.numpy()
    base = np.arctan2(u[:, 1], u[:, 0]) / 4.0
    bxy = st[boundary_ids]

    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    for k in range(4):
        ang = base[boundary_ids] + k * (np.pi / 2)
        ax.quiver(bxy[:, 0], bxy[:, 1], np.cos(ang), np.sin(ang),
                  color="#3aa6d0", scale=18, width=0.004,
                  headwidth=0, headlength=0, pivot="mid", alpha=0.85)
    ax.quiver(bxy[:, 0], bxy[:, 1],
              np.cos(base[boundary_ids]), np.sin(base[boundary_ids]),
              color="red", scale=18, width=0.005,
              headwidth=4, headlength=5, pivot="tail", alpha=1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_mapping(out_png):
    """step04 companion: conceptual cross -> 4*theta representative mapping."""
    theta = np.deg2rad(20.0)
    rep = 4.0 * theta
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.75", lw=1.0, zorder=1)
    for k in range(4):
        ang = theta + k * (np.pi / 2)
        ax.quiver(0.0, 0.0, np.cos(ang), np.sin(ang),
                  color="#1f77ff", scale=2.4, width=0.012,
                  headwidth=4, headlength=5, pivot="tail", zorder=3,
                  alpha=0.95)
    ax.quiver(0.0, 0.0, np.cos(rep), np.sin(rep),
              color="red", scale=2.4, width=0.014,
              headwidth=4, headlength=5, pivot="tail", zorder=4)
    ax.text(0.62 * np.cos(theta / 2), 0.62 * np.sin(theta / 2),
            r"$\theta$", color="0.35", fontsize=12, ha="center", va="center")
    ax.text(np.cos(rep) - 0.22, np.sin(rep) + 0.02,
            r"$4\theta$", color="red", fontsize=12, ha="right", va="center")
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_framefield(mesh, st, tris, loops, out_vectors, out_crosses):
    """step05: smoothed representation vector field and the back-mapped
    4-RoSy frame field on every node."""
    u = mesh.u.numpy()
    base = np.arctan2(u[:, 1], u[:, 0]) / 4.0

    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    ax.quiver(st[:, 0], st[:, 1], u[:, 0], u[:, 1],
              color="#87CEEB", scale=26, width=0.0022,
              headwidth=0, headlength=0, pivot="mid", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_vectors)

    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    for k in range(4):
        ang = base + k * (np.pi / 2)
        ax.quiver(st[:, 0], st[:, 1], np.cos(ang), np.sin(ang),
                  color="#87CEEB", scale=26, width=0.0022,
                  headwidth=0, headlength=0, pivot="mid", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_crosses)


def showcase_singularities(mesh, st, tris, loops, final_sing_st, out_png):
    """step06: frame field with the FINAL partition singularities marked
    (red rings). ``final_sing_st`` are positions already in raw (s, t)."""
    u = mesh.u.numpy()
    base = np.arctan2(u[:, 1], u[:, 0]) / 4.0

    fig, ax = plt.subplots(figsize=(4, 6))
    _draw_domain(ax, st, tris, loops)
    for k in range(4):
        ang = base + k * (np.pi / 2)
        ax.quiver(st[:, 0], st[:, 1], np.cos(ang), np.sin(ang),
                  color="#87CEEB", scale=40, width=0.0015,
                  headwidth=0, headlength=0, pivot="mid", alpha=0.9)
    if len(final_sing_st):
        ax.scatter(final_sing_st[:, 0], final_sing_st[:, 1],
                   facecolors="none", edgecolors="red", s=150,
                   linewidths=2.0, zorder=6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig, out_png)


def _ends_ok_fn(mesh):
    """Predicate: does a curve endpoint land on a legitimate target (boundary
    or termination node), judged modulo the pitch?"""
    pitch = mesh.pitch_norm
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    boundary = [np.asarray(b, float) for b in mesh.streamlines[:n_b]]
    term = ps._termination_nodes(mesh)

    def _ends_ok(p):
        for k in range(-ps.MAX_WRAPS, ps.MAX_WRAPS + 1):
            q = np.asarray(p, float) - [k * pitch, 0.0]
            if len(term) and np.min(np.linalg.norm(term - q, axis=1)) < 0.05:
                return True
            if ps._min_boundary_dist(q, boundary) < 0.05:
                return True
        return False

    return _ends_ok


def _fold_boundary(ax, s, pitch, s_left, ring_tol=0.02, **kw):
    """Boundary curve: closed curves (winding orbits, blade outline) are
    seam-folded; open wall segments (seams, axial walls) are drawn as-is so
    the domain outline stays true (folding a wall would shift it off-domain)."""
    if np.linalg.norm(s[0] - s[-1]) < ring_tol:
        for seg in fold_curve(s, pitch, s_left):
            ax.plot(seg[:, 0], seg[:, 1], **kw)
    else:
        ax.plot(s[:, 0], s[:, 1], **kw)


def showcase_integration(mesh, out_png, aspect):
    """step07: raw streamline integration state (pre-snap, full spirals),
    folded into the fundamental pitch."""
    pitch = mesh.pitch_norm
    s_left = _seam_left_fn(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    _ends_ok = _ends_ok_fn(mesh)
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")
    cmap = plt.cm.tab20
    for i, s in enumerate(mesh.streamlines):
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) < 2:
            continue
        if i < n_b:
            _fold_boundary(ax, s, pitch, s_left, color="black", lw=1.0,
                           zorder=2)
        else:
            color = cmap(i % 20 / 20.0)
            segs = _streamline_segs(s, pitch, s_left, _ends_ok=_ends_ok,
                                    trim_trailing_partial=True)
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.0,
                        alpha=1.0, zorder=5)
    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


def _place_labels(ax, items, min_sep=0.03, zorder=6):
    """Draw index labels while avoiding overlap: a label colliding with an
    already-placed one is stepped outward on a ring until clear, with a thin
    leader line back to its anchor. Non-colliding labels stay put."""
    placed = []
    for anchor, text, color in items:
        pos = np.asarray(anchor, float)
        if any(np.linalg.norm(pos - p) < min_sep for p in placed):
            for r in np.arange(min_sep, 6 * min_sep + 1e-9, min_sep):
                cand = None
                for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    c = pos + r * np.array([np.cos(a), np.sin(a)])
                    if all(np.linalg.norm(c - p) >= min_sep for p in placed):
                        cand = c
                        break
                if cand is not None:
                    pos = cand
                    break
        if not np.allclose(pos, anchor):
            ax.plot([anchor[0], pos[0]], [anchor[1], pos[1]], lw=0.4,
                    color="0.6", zorder=zorder - 1)
        ax.text(pos[0], pos[1], text, fontsize=7, color=color, ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.15",
                facecolor="white", edgecolor="none", alpha=0.7), zorder=zorder)
        placed.append(pos)


def showcase_integration_labeled(mesh, out_png, aspect):
    """step07 labeled: every curve in its own color with an index label on
    its longest folded segment; singularities as color-coded stars with
    'S1 (x, y)' legend entries instead of in-plot text boxes."""
    pitch = mesh.pitch_norm
    s_left = _seam_left_fn(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    _ends_ok = _ends_ok_fn(mesh)
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")

    cmap = plt.cm.tab20
    labels = []
    for i, s in enumerate(mesh.streamlines):
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) < 2:
            continue
        color = cmap(i % 20 / 20.0) if i >= n_b else "black"
        lw = 2.0 if i >= n_b else 1.0
        zorder = 5 if i >= n_b else 2
        # open boundary walls stay as-is (folding would shift them off-domain);
        # separatrices and closed orbits are seam-folded
        if i < n_b and np.linalg.norm(s[0] - s[-1]) >= 0.02:
            segs = [s]
        else:
            if i >= n_b:
                segs = _streamline_segs(s, pitch, s_left, _ends_ok=_ends_ok,
                                        trim_trailing_partial=True)
            else:
                segs = fold_curve(s, pitch, s_left)
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=zorder)
        if segs:
            longest = max(segs, key=len)
            mid = longest[len(longest) // 2]
            labels.append((mid, str(i), "black" if i >= n_b else "0.4"))
    _place_labels(ax, labels, zorder=6)

    sing_cmap = plt.cm.tab10
    for si, (name, data) in enumerate(get_singularities(mesh).items()):
        c = data["coords"]
        ax.scatter(c[0], c[1], c=[sing_cmap(si % 10)], s=180, marker="*",
                   zorder=8, edgecolors="black", linewidths=0.8,
                   label=f"{name} ({c[0]:.3f}, {c[1]:.3f})")

    ax.plot([], [], "-", color="black", lw=1.0, label="boundary")
    ax.plot([], [], "-", color="0.3", lw=2.0, label="separatrix")
    ax.set_aspect(aspect)
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, out_png)


def showcase_separatrices_per_singularity(mesh, out_dir, aspect, part="",
                                           tol=0.03, streamlines=None):
    """One labeled plot per singularity: every separatrix incident on it --
    outgoing (originating there, solid) and incoming (terminating there,
    dashed) -- seam-folded, colored per arm with index labels, over the faint
    domain and boundary. Star marks the singularity; the legend lists which
    indices are outgoing and which incoming. Written to
    ``out_dir/S{k}_separatrices.png``.

    Separatrices come from ``mesh.streamlines`` by default; pass
    ``streamlines`` (e.g. StreamlineMerging.new_streamlines) to plot that
    list instead, with incidence judged by endpoint proximity."""
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pitch = mesh.pitch_norm
    s_left = _seam_left_fn(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    _ends_ok = _ends_ok_fn(mesh)
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()
    sings = get_singularities(mesh)
    sing_cmap = plt.cm.tab10
    arm_cmap = plt.cm.tab20

    # separatrix endpoints (cover coords) to find arms terminating at a
    # singularity, judged modulo the pitch since an arm may dock across a seam
    def _near(p, q):
        for k in range(-ps.MAX_WRAPS, ps.MAX_WRAPS + 1):
            if np.linalg.norm((np.asarray(p, float) - [k * pitch, 0.0]) - q) \
                    < tol:
                return True
        return False

    if streamlines is not None:
        curves = {i: np.asarray(s, float) for i, s in enumerate(streamlines)}
    else:
        curves = {n_b + i: np.asarray(mesh.streamlines[n_b + i], float)
                  for i in range(len(mesh.separatrices))}

    for si, (name, data) in enumerate(sings.items()):
        c = data["coords"]
        if streamlines is not None:
            outgoing = [i for i, s in curves.items()
                        if s.ndim == 2 and len(s) >= 2 and _near(s[0], c)]
            incoming = [i for i, s in curves.items()
                        if s.ndim == 2 and len(s) >= 2
                        and i not in outgoing and _near(s[-1], c)]
        else:
            outgoing = list(data["sep_indices"])
            incoming = []
            for idx, s in curves.items():
                if idx in outgoing:
                    continue
                if s.ndim != 2 or len(s) < 2:
                    continue
                if _near(s[-1], c) or _near(s[0], c):
                    incoming.append(idx)

        fig, ax = plt.subplots(figsize=(6, 9))
        ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")
        for i in range(n_b):
            b = np.asarray(mesh.streamlines[i], float)
            if b.ndim == 2 and len(b) >= 2:
                _fold_boundary(ax, b, pitch, s_left, color="0.75", lw=0.6,
                               alpha=0.6, zorder=1)

        labels = []
        for group, style, trim in ((outgoing, "-", True),
                                   (incoming, (0, (4, 2)), False)):
            for j, idx in enumerate(group):
                s = curves[idx]
                if s.ndim != 2 or len(s) < 2:
                    continue
                color = arm_cmap(j % 20 / 20.0)
                segs = fold_curve(s, pitch, s_left)
                if trim and segs and not _ends_ok(s[-1]):
                    segs = segs[:-1]
                for seg in segs:
                    ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.0,
                            ls=style, zorder=5)
                if segs:
                    longest = max(segs, key=len)
                    labels.append((longest[len(longest) // 2], str(idx),
                                   "black"))
        _place_labels(ax, labels, zorder=6)

        ax.scatter(c[0], c[1], c=[sing_cmap(si % 10)], s=220, marker="*",
                   zorder=8, edgecolors="black", linewidths=0.9,
                   label=f"{name} ({c[0]:.3f}, {c[1]:.3f})")
        out_txt = ", ".join(str(i) for i in outgoing) or "none"
        in_txt = ", ".join(str(i) for i in incoming) or "none"
        handles = [Line2D([], [], color="0.3", lw=2.0, ls="-",
                          label=f"outgoing: {out_txt}"),
                   Line2D([], [], color="0.3", lw=2.0, ls=(0, (4, 2)),
                          label=f"incoming: {in_txt}")]
        ax.set_aspect(aspect)
        ax.set_axis_off()
        h0, l0 = ax.get_legend_handles_labels()
        ax.legend(h0 + handles, l0 + [h.get_label() for h in handles],
                  loc="upper left", fontsize=8)
        stem = f"{part}_{name}" if part else name
        _save(fig, out_dir / f"{stem}_separatrices.png")


# --------------------------------------------------------------------------
# steps 8-11: per method
# --------------------------------------------------------------------------

def _singularity_legend(ax, mesh):
    """Color-coded singularity stars with 'S1 (x, y)' legend entries."""
    sing_cmap = plt.cm.tab10
    for si, (name, data) in enumerate(get_singularities(mesh).items()):
        c = data["coords"]
        ax.scatter(c[0], c[1], c=[sing_cmap(si % 10)], s=180, marker="*",
                   zorder=8, edgecolors="black", linewidths=0.8,
                   label=f"{name} ({c[0]:.3f}, {c[1]:.3f})")


def showcase_simplification(mesh, merged_streamlines, wedge_arms, out_png,
                            aspect):
    """step08: streamlines after the Xiao merge (Alg. 2 cases 1-3). For ta/tb
    the wedge arms that the seam postprocessing deletes are shown dashed
    orange; pass an empty ``wedge_arms`` for xiao."""
    pitch = mesh.pitch_norm
    s_left = _seam_left_fn(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    boundary = [np.asarray(b, float) for b in mesh.streamlines[:n_b]]
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")
    cmap = plt.cm.tab20
    deduped = _dedup_pitch_shifted(merged_streamlines, pitch)
    for i, s in enumerate(deduped):
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) < 2:
            continue
        if _on_seam(s, pitch, s_left):
            continue
        color = cmap(i % 20 / 20.0)
        for seg in _streamline_segs(s, pitch, s_left):
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=1.6, zorder=5)
    for b in boundary:
        b = np.asarray(b, float)
        if b.ndim != 2 or len(b) < 2:
            continue
        # drawn above the streamlines so seams read as one clean black line
        # even where a streamline legitimately runs along them
        _fold_boundary(ax, b, pitch, s_left, color="black", lw=1.5, zorder=6)
    for arm in wedge_arms:
        arm = np.asarray(arm, float)
        if arm.ndim != 2 or len(arm) < 2:
            continue
        for seg in fold_curve(arm, pitch, s_left):
            ax.plot(seg[:, 0], seg[:, 1], ":", color="red",
                    lw=2.0, zorder=7)
    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_simplification_labeled(mesh, merged_streamlines, wedge_arms,
                                    out_png, aspect):
    """step08 labeled: each merged streamline in its own color with an index
    label; boundary walls as-is, deleted wedge arms dashed orange; color-coded
    singularity stars with legend."""
    pitch = mesh.pitch_norm
    s_left = _seam_left_fn(mesh)
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    boundary = [np.asarray(b, float) for b in mesh.streamlines[:n_b]]
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")
    for b in boundary:
        b = np.asarray(b, float)
        if b.ndim != 2 or len(b) < 2:
            continue
        _fold_boundary(ax, b, pitch, s_left, color="black", lw=1.0, zorder=2)

    cmap = plt.cm.tab20
    labels = []
    deduped = _dedup_pitch_shifted(merged_streamlines, pitch)
    for i, s in enumerate(deduped):
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) < 2:
            continue
        if _on_seam(s, pitch, s_left):
            continue
        color = cmap(i % 20 / 20.0)
        segs = _streamline_segs(s, pitch, s_left)
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=1.6, zorder=5)
        if segs:
            longest = max(segs, key=len)
            mid = longest[len(longest) // 2]
            labels.append((mid, str(i), "black"))
    _place_labels(ax, labels, zorder=6)

    for arm in wedge_arms:
        arm = np.asarray(arm, float)
        if arm.ndim != 2 or len(arm) < 2:
            continue
        for seg in fold_curve(arm, pitch, s_left):
            ax.plot(seg[:, 0], seg[:, 1], "--", color="darkorange",
                    lw=2.0, zorder=7)
    if len(wedge_arms):
        ax.plot([], [], "--", color="darkorange", lw=2.0,
                label="deleted wedge arm")

    _singularity_legend(ax, mesh)
    ax.set_aspect(aspect)
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, out_png)


def _postproc_data(run):
    """Blocks, boundary_ref, irregular interior nodes and deleted wedge arms
    for the final partition of one run."""
    result = run["result"]
    mesh = run["mesh"]
    boundary_ref = run["boundary_ref"]
    wedge_arms = getattr(mesh, "dropped_wedge_arms", []) or []
    _r, _t, irregular = tmf.node_regularity(
        result, lambda p: ps._min_boundary_dist(p, boundary_ref))
    return result, mesh, boundary_ref, irregular, wedge_arms


def showcase_postprocessing(run, out_png, aspect):
    """step09: final block structure of one method; wedge streamlines deleted
    at the seam (ta/tb) drawn red with their source singularities ringed."""
    result, mesh, boundary_ref, irregular, wedge_arms = _postproc_data(run)
    nodes = result["nodes"]

    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")

    cmap = plt.cm.tab20
    for i, blk in enumerate(result["blocks"]):
        ring = blk["ring"]
        color = cmap(i % 20 / 20.0)
        ax.fill(ring[:, 0], ring[:, 1], alpha=0.25, color=color)
        for s in blk["sides"]:
            ax.plot(s[:, 0], s[:, 1], color=color, lw=1.1, zorder=3)
        c = nodes[blk["corners"]]
        ax.scatter(c[:, 0], c[:, 1], c="black", s=10, zorder=5)

    for b in boundary_ref:
        b = np.asarray(b, float)
        if b.ndim == 2 and len(b) >= 2:
            ax.plot(b[:, 0], b[:, 1], color="black", lw=1.6, zorder=4)

    sing_xy = nodes[irregular] if irregular else np.zeros((0, 2))
    ring_sings = set()
    for arm in wedge_arms:
        arm = np.asarray(arm, float)
        if arm.ndim != 2 or len(arm) < 2:
            continue
        ax.plot(arm[:, 0], arm[:, 1], color="red", lw=2.2, zorder=8)
        if len(sing_xy):
            d0 = np.min(np.linalg.norm(sing_xy - arm[0], axis=1))
            d1 = np.min(np.linalg.norm(sing_xy - arm[-1], axis=1))
            tip = arm[0] if d0 <= d1 else arm[-1]
            ring_sings.add(
                int(np.argmin(np.linalg.norm(sing_xy - tip, axis=1))))
    for si in ring_sings:
        ax.scatter(sing_xy[si, 0], sing_xy[si, 1], facecolors="none",
                   edgecolors="red", s=170, linewidths=2.2, zorder=9)

    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_postprocessing_labeled(run, out_png, aspect):
    """step09 labeled: final blocks numbered at their centroid; irregular
    interior nodes (valence != 4) ringed red with a legend entry."""
    result, mesh, boundary_ref, irregular, _wedge = _postproc_data(run)
    nodes = result["nodes"]

    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="black")

    cmap = plt.cm.tab20
    for i, blk in enumerate(result["blocks"]):
        ring = blk["ring"]
        ax.fill(ring[:, 0], ring[:, 1], alpha=0.25, color=cmap(i % 20 / 20.0))
        for s in blk["sides"]:
            ax.plot(s[:, 0], s[:, 1], color="0.3", lw=1.0, zorder=3)
        c = nodes[blk["corners"]]
        ax.scatter(c[:, 0], c[:, 1], c="black", s=10, zorder=5)
        cen = ring[:-1].mean(axis=0) if len(ring) > 1 else ring.mean(axis=0)
        ax.text(cen[0], cen[1], str(i), fontsize=9, fontweight="bold",
                color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.75), zorder=7)

    for b in boundary_ref:
        b = np.asarray(b, float)
        if b.ndim == 2 and len(b) >= 2:
            ax.plot(b[:, 0], b[:, 1], color="black", lw=1.6, zorder=4)

    if irregular:
        ax.scatter(nodes[irregular, 0], nodes[irregular, 1],
                   facecolors="none", edgecolors="red", s=150, linewidths=2.0,
                   zorder=9, label=f"irregular interior ({len(irregular)})")
        ax.legend(loc="upper left", fontsize=8)

    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_tfi(tfi, result, out_png, aspect):
    """step10: TFI grid (ta/tb only)."""
    fig, ax = plt.subplots(figsize=(4, 6))
    cmap = plt.cm.tab20
    for i, blk in enumerate(result["blocks"]):
        ring = blk["ring"]
        color = cmap(i % 20 / 20.0)
        ax.fill(ring[:, 0], ring[:, 1], alpha=0.25, color=color)
    for X in tfi["grids"]:
        for i in range(X.shape[0]):
            ax.plot(X[i, :, 0], X[i, :, 1], "0.4", lw=0.3)
        for j in range(X.shape[1]):
            ax.plot(X[:, j, 0], X[:, j, 1], "0.4", lw=0.3)
    for blk in result["blocks"]:
        for s in blk["sides"]:
            ax.plot(s[:, 0], s[:, 1], "black", lw=1.0)
    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


def showcase_tiled(result, tfi, pitch, out_png, aspect):
    """step11: TFI grid tiled over 3 passages (shifts -pitch, 0, +pitch), the
    visual periodicity check -- grid points of neighbouring copies must
    coincide at the interior seams."""
    shifts = [-pitch, 0.0, pitch]
    cols = ["0.55", "black", "0.55"]

    fig, ax = plt.subplots(figsize=(18, 7))
    for sh, col in zip(shifts, cols):
        for X in tfi["grids"]:
            for i in range(X.shape[0]):
                ax.plot(X[i, :, 0] + sh, X[i, :, 1], col, lw=0.25)
            for j in range(X.shape[1]):
                ax.plot(X[:, j, 0] + sh, X[:, j, 1], col, lw=0.25)
        for blk in result["blocks"]:
            for s in blk["sides"]:
                ax.plot(s[:, 0] + sh, s[:, 1], col, lw=0.8)
    ax.set_aspect(aspect)
    ax.set_axis_off()
    _save(fig, out_png)


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def write_plots(stl, part, runs, out_dir):
    """Write the full showcase series for one surface into ``out_dir``.

    ``runs`` maps method name (ta/tb/xiao) to its completed run dict from
    tmesh.run_tmesh / xiao.run_xiao. Steps 1-7 are written once per part,
    steps 8-11 per method. Assumes the module globals (periodic field,
    corner modes) are still configured like the runs.
    """
    from .dp_adapter import build_dp_data
    from .field import FrameField, StreamlineGenerator_v2
    from .field.singularity_detector import detect_singularities
    from .field.streamline_merging import StreamlineMerging
    from .unwrap_surface import unwrap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    showcase_3d_surface(stl, out_dir / f"step01_3d_surface_{part}.png")

    data = unwrap(stl)
    st, tris, loops = data["st"], data["tris"], data["loops"]
    showcase_unwrapped(st, tris, loops,
                       out_dir / f"step02_unwrapped_{part}.png")

    # fresh field state, same config as the runs
    mesh, transform = build_dp_data(stl)
    ff = FrameField(mesh)
    detect_singularities(ff.mesh)

    # steps 7-11 draw in normalized [0,1]^2; the normalization is a pure
    # per-axis scale, so this aspect renders the true (s,t) shape of steps 1-6
    aspect = ((transform["tmax"] - transform["tmin"]) /
              (transform["smax"] - transform["smin"]))

    showcase_crossfield(mesh, st, tris, loops,
                        out_dir / f"step03_crossfield_{part}.png")
    showcase_representatives(mesh, st, tris, loops,
                             out_dir / f"step04_representatives_{part}.png")
    showcase_mapping(out_dir / "step04_mapping.png")
    showcase_framefield(mesh, st, tris, loops,
                        out_dir / f"step05_vectorfield_{part}.png",
                        out_dir / f"step05_framefield_{part}.png")

    # final partition singularities (identical across methods), mapped
    # normalized [0,1]^2 -> raw (s,t): both frames share node ordering, so a
    # per-axis linear fit recovers the map
    ref = next((runs[m] for m in ("ta", "tb", "xiao") if m in runs), None)
    final_sing_st = np.zeros((0, 2))
    if ref is not None:
        sings = get_singularities(ref["mesh"])
        if sings:
            pos_norm = np.array([d["coords"] for d in sings.values()])
            mx = mesh.x[:, 0:2].numpy()
            cx = np.polyfit(mx[:, 0], st[:, 0], 1)
            cy = np.polyfit(mx[:, 1], st[:, 1], 1)
            final_sing_st = np.column_stack(
                [np.polyval(cx, pos_norm[:, 0]),
                 np.polyval(cy, pos_norm[:, 1])])
    showcase_singularities(mesh, st, tris, loops, final_sing_st,
                           out_dir / f"step06_singularities_{part}.png")

    # raw integration state (pre-snap), then the Xiao-merged state
    sl = StreamlineGenerator_v2(ff.mesh)
    ps._drop_degenerate_corner_seps(sl.mesh)
    showcase_integration(
        sl.mesh, out_dir / f"step07_streamline_integration_{part}.png",
        aspect)
    showcase_integration_labeled(
        sl.mesh,
        out_dir / f"step07_streamline_integration_{part}_labeled.png",
        aspect)

    ps._snap_separatrix_endpoints(sl.mesh, radius=0.045)
    showcase_separatrices_per_singularity(
        sl.mesh, out_dir / "separatrices", aspect, part=part)
    merging = StreamlineMerging(sl.mesh, verbose=False)
    merged = [np.asarray(s, float) for s in merging.new_streamlines]

    for method, run in runs.items():
        wedge_arms = getattr(run["mesh"], "dropped_wedge_arms", []) or []
        showcase_separatrices_per_singularity(
            run["mesh"], out_dir / "separatrices", aspect,
            part=f"{part}_{method}_post",
            streamlines=run["merging"].new_streamlines)
        showcase_simplification(
            sl.mesh, merged, wedge_arms,
            out_dir / f"step08_simplification_{part}_{method}.png", aspect)
        showcase_simplification_labeled(
            sl.mesh, merged, wedge_arms,
            out_dir / f"step08_simplification_{part}_{method}_labeled.png",
            aspect)
        showcase_postprocessing(
            run, out_dir / f"step09_postprocessing_{part}_{method}.png",
            aspect)
        showcase_postprocessing_labeled(
            run, out_dir / f"step09_postprocessing_{part}_{method}_labeled.png",
            aspect)
        tfi = run.get("tfi")
        if tfi and tfi["grids"]:
            showcase_tfi(tfi, run["result"],
                         out_dir / f"step10_tfi_{part}_{method}.png", aspect)
            showcase_tiled(run["result"], tfi, float(run["mesh"].pitch_norm),
                           out_dir / f"step11_tiled_{part}_{method}.png",
                           aspect)
