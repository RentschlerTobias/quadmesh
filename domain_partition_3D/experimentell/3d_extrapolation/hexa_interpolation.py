#!/usr/bin/env python3
"""Automatic 3D hexa domain partition: hub template -> shroud morph -> lift.

The cross fields of hub and shroud are topologically DIFFERENT (hub: four
idx=-1 singularities, shroud: four idx=+1 -- physical, blade twist; verified
with isotropic normalization). A spanwise hexa partition needs ONE common
block topology, so the validated hub T-a block structure is used as the
template and morphed onto the shroud surface:

  1. hub T-a run (14 blocks, conforming TFI) = master template
  2. morph hub-normalized skeleton -> shroud-normalized coordinates:
     - affine parallelogram map (exact on the outer boundary: seams map to
       seams incl. pitch, inlet/outlet to inlet/outlet)
     - blade correspondence by arclength (anchored at LE/TE = farthest pair):
       points ON the hub blade land exactly ON the shroud blade
     - interior: inverse-distance blend of the blade displacement, decaying
       to zero at the outer boundary
  3. shroud TFI with the SAME edge divisions (identical (n_u, n_v) per block)
  4. hexa blocks: X(u,v,w) = (1-w)*Hub3D(u,v) + w*Shroud3D(u,v) with a
     tanh-clustered w-distribution (hub/shroud are walls), inverse cylinder
     unwrap per surface (theta = s/r, z = t)
  5. export: VTK (hexahedra) + plots; gates: 0 inverted TFI cells on the
     morphed shroud, 0 non-positive hexa cell volumes, matching grid
     dimensions per block pair.

Outputs to output/T1_9/hexa_3d/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dp3d import unwrap_surface as us                                    # noqa: E402
from dp3d import clean_separatrix as cs                                  # noqa: E402
from dp3d import partition_surface as ps                                 # noqa: E402
from dp3d import tmesh_faces as tmf                                      # noqa: E402
from dp3d import tmesh as tp                                   # noqa: E402
from dp3d.dp_adapter import build_dp_data                           # noqa: E402

HUB_STL = "/root/repos/block_structured_meshing/T1_9_hub_raw.stl"
SHROUD_STL = "/root/repos/block_structured_meshing/T1_9_shroud_raw.stl"
OUT = Path("/root/repos/block_structured_meshing/output/T1_9/hexa_3d")

N_SPAN = 8            # spanwise cells hub->shroud
SPAN_RATIO = 5.0      # tanh clustering to both walls


# --------------------------------------------------------------------------
# blade correspondence + morph
# --------------------------------------------------------------------------

def _loop_resample(loop, n=400):
    """Closed loop -> n arclength-uniform points, CCW, starting at the LE
    (farthest-pair end with smaller t)."""
    loop = np.asarray(loop, float)
    if np.linalg.norm(loop[0] - loop[-1]) > 1e-12:
        loop = np.vstack([loop, loop[0]])
    if tmf._shoelace(loop) < 0:
        loop = loop[::-1]
    # farthest pair = LE/TE chord
    sub = loop[:: max(1, len(loop) // 64)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    a, b = sub[i], sub[j]
    le = a if a[1] < b[1] else b
    k0 = int(np.argmin(np.linalg.norm(loop - le, axis=1)))
    loop = np.vstack([loop[k0:-1], loop[:k0], loop[k0:k0 + 1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.linspace(0.0, cum[-1], n + 1)[:-1]
    return np.column_stack([np.interp(ss, cum, loop[:, 0]),
                            np.interp(ss, cum, loop[:, 1])]), cum[-1]


class HubToShroudMorph:
    """Map hub-normalized [0,1]^2 coordinates to shroud-normalized ones."""

    def __init__(self, hub_mesh, shroud_mesh, n_blade=400):
        self.slope_h = self._seam_slope(hub_mesh)
        self.slope_s = self._seam_slope(shroud_mesh)
        self.p_h = float(hub_mesh.pitch_norm)
        self.p_s = float(shroud_mesh.pitch_norm)
        bh, self.len_h = _loop_resample(hub_mesh.blade_loops[0], n_blade)
        bs, self.len_s = _loop_resample(shroud_mesh.blade_loops[0], n_blade)
        self.blade_h = bh
        self.blade_s = bs
        self.blade_h_aff = self.affine(bh)
        self.delta = bs - self.blade_h_aff          # blade displacement field
        # affine images of the ACTUAL seam walls (the physical seams are not
        # perfectly straight; a line-based outer distance left fade > 0 ON
        # the seam and broke L/R conformity)
        A, B = tp._periodic_chains(hub_mesh)
        self.wall_l = self.affine(A)
        self.wall_r = self.affine(B)

    @staticmethod
    def _seam_slope(mesh):
        A, B = tp._periodic_chains(mesh)
        return float((A[-1, 0] - A[0, 0]) / max(A[-1, 1] - A[0, 1], 1e-12))

    def affine(self, pts):
        """Parallelogram->parallelogram: u' = u * p_s/p_h in seam-aligned
        coords (exact on the whole outer boundary)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        u = pts[:, 0] - self.slope_h * pts[:, 1]
        s = u * (self.p_s / self.p_h) + self.slope_s * pts[:, 1]
        return np.column_stack([s, pts[:, 1]])

    def _outer_dist(self, pts):
        """Distance to the outer boundary (t=0, t=1, actual seam walls) in
        affine-image coordinates."""
        pts = np.atleast_2d(pts)
        d = np.minimum(pts[:, 1], 1.0 - pts[:, 1])
        for i, p in enumerate(pts):
            d[i] = min(d[i], tmf._dist_to_polyline(p, self.wall_l),
                       tmf._dist_to_polyline(p, self.wall_r))
        return d

    def __call__(self, pts, fade_dist=0.15):
        pts = np.atleast_2d(np.asarray(pts, float))
        out = self.affine(pts)
        # exact arclength mapping for points ON the hub blade
        d_blade = np.linalg.norm(
            pts[:, None, :] - self.blade_h[None, :, :], axis=2)
        nearest = d_blade.argmin(axis=1)
        on_blade = d_blade[np.arange(len(pts)), nearest] < 5e-4
        # IDW interpolation of the blade displacement, multiplied by a
        # SMOOTH fade to zero at the outer boundary. (A competitive
        # zero-anchor weight 1/d_out^2 is not monotone along a curve running
        # into the seam and produced spikes there.)
        w = 1.0 / (np.linalg.norm(
            out[:, None, :] - self.blade_h_aff[None, :, :], axis=2) ** 2
            + 1e-12)
        disp = (w @ self.delta) / w.sum(axis=1)[:, None]
        x = np.clip(self._outer_dist(out) / fade_dist, 0.0, 1.0)
        fade = x * x * (3.0 - 2.0 * x)               # C1 smoothstep
        out = out + disp * fade[:, None]
        if on_blade.any():
            out[on_blade] = self.blade_s[nearest[on_blade]]
        return out


def morph_result(hub_result, morph):
    """Morph the whole hub block skeleton (nodes, e2s polylines, sides,
    rings) into shroud coordinates; topology (ids/chains) is untouched."""
    import copy
    res = {"planar": True, "n_parallel": hub_result["n_parallel"]}
    res["nodes"] = morph(hub_result["nodes"])
    res["e2s"] = {k: morph(poly) for k, poly in hub_result["e2s"].items()}
    res["edges"] = list(hub_result["edges"])
    blocks = []
    for blk in hub_result["blocks"]:
        b = {"cycle": list(blk["cycle"]),
             "corner_pos": list(blk["corner_pos"]),
             "corners": list(blk["corners"]),
             "side_chains": [list(c) for c in blk["side_chains"]],
             "sides": [morph(s) for s in blk["sides"]],
             "ring": morph(blk["ring"])}
        b["area"] = tmf._shoelace(b["ring"])
        b["centroid"] = b["ring"][:-1].mean(axis=0)
        blocks.append(b)
    res["blocks"] = blocks
    res["rejects"] = []
    res["blade_idx"] = list(hub_result["blade_idx"])
    res["blade_regions"] = copy.deepcopy(hub_result["blade_regions"])
    return res


# --------------------------------------------------------------------------
# 3D lift + hexa assembly
# --------------------------------------------------------------------------

def to_cyl(pts, transform):
    """Normalized [0,1]^2 -> cylinder coords (theta, z) with theta = s/r."""
    pts = np.atleast_2d(np.asarray(pts, float))
    s = pts[:, 0] * (transform["smax"] - transform["smin"]) + transform["smin"]
    t = pts[:, 1] * (transform["tmax"] - transform["tmin"]) + transform["tmin"]
    return np.column_stack([s / transform["r"], t])


def lift_to_3d(pts, transform):
    """Normalized [0,1]^2 -> (s,t) -> cylinder surface (theta = s/r, z = t)."""
    tz = to_cyl(pts, transform)
    r = transform["r"]
    return np.column_stack([r * np.cos(tz[:, 0]), r * np.sin(tz[:, 0]),
                            tz[:, 1]])


def span_fractions(n=N_SPAN, ratio=SPAN_RATIO):
    return tp.edge_fractions(n, True, True, ratio=ratio)


def build_hexa_blocks(grids_hub, grids_shroud, tf_hub, tf_shroud,
                      n_span=N_SPAN):
    """Per block: (n_v, n_u, n_w+1, 3) hexa grid. Interpolation is done in
    CYLINDER coordinates (r, theta, z linear in w): a cartesian ruled blend
    between r=0.5 and r=1.9 shears where hub and shroud points are
    theta-offset (blade twist) and flips thin wall cells; the cylindrical
    blend turns the twist into a clean helical ruling."""
    w = span_fractions(n_span)
    r_h, r_s = tf_hub["r"], tf_shroud["r"]
    hexa = []
    for Gh, Gs in zip(grids_hub, grids_shroud):
        assert Gh.shape == Gs.shape, "hub/shroud grid dims differ"
        nv, nu, _ = Gh.shape
        TH = to_cyl(Gh.reshape(-1, 2), tf_hub).reshape(nv, nu, 2)
        TS = to_cyl(Gs.reshape(-1, 2), tf_shroud).reshape(nv, nu, 2)
        r = (1.0 - w) * r_h + w * r_s                       # (nw+1,)
        tz = ((1.0 - w)[None, None, :, None] * TH[:, :, None, :]
              + w[None, None, :, None] * TS[:, :, None, :])  # theta, z
        X = np.stack([r[None, None, :] * np.cos(tz[..., 0]),
                      r[None, None, :] * np.sin(tz[..., 0]),
                      tz[..., 1]], axis=-1)
        hexa.append(X)
    return hexa


def hexa_cell_volumes(X):
    """Signed volumes of all cells of one block grid (nv,nu,nw,3), via a
    5-tetrahedron decomposition (used for volume statistics only -- a valid
    strongly sheared hexahedron can have individual inverted tets, so cell
    VALIDITY is judged by corner Jacobians, see hexa_corner_jacobians)."""
    c = _cell_corners(X)

    def tet(a, b, cc, d):
        return np.einsum("...i,...i->...", b - a,
                         np.cross(cc - a, d - a)) / 6.0

    return (tet(c[0], c[1], c[2], c[5]) + tet(c[0], c[2], c[3], c[7])
            + tet(c[0], c[5], c[4], c[7]) + tet(c[2], c[7], c[6], c[5])
            + tet(c[0], c[2], c[7], c[5]))


def _cell_corners(X):
    return [X[:-1, :-1, :-1], X[:-1, 1:, :-1], X[1:, 1:, :-1],
            X[1:, :-1, :-1], X[:-1, :-1, 1:], X[:-1, 1:, 1:],
            X[1:, 1:, 1:], X[1:, :-1, 1:]]


def hexa_corner_jacobians(X):
    """Minimum corner Jacobian per cell (trilinear hexa validity: cell is
    valid iff all 8 corner determinants det(e_u, e_v, e_w) share the grid
    orientation sign)."""
    c = _cell_corners(X)

    def det(eu, ev, ew):
        return np.einsum("...i,...i->...", eu, np.cross(ev, ew))

    J = np.stack([
        det(c[1] - c[0], c[3] - c[0], c[4] - c[0]),
        det(c[1] - c[0], c[2] - c[1], c[5] - c[1]),
        det(c[2] - c[3], c[2] - c[1], c[6] - c[2]),
        det(c[2] - c[3], c[3] - c[0], c[7] - c[3]),
        det(c[5] - c[4], c[7] - c[4], c[4] - c[0]),
        det(c[5] - c[4], c[6] - c[5], c[5] - c[1]),
        det(c[6] - c[7], c[6] - c[5], c[6] - c[2]),
        det(c[6] - c[7], c[7] - c[4], c[7] - c[3]),
    ], axis=0)
    return J.min(axis=0), J.max(axis=0)


def export_vtk(hexa, out_path):
    import meshio
    pts_all, cells_all, off = [], [], 0
    for X in hexa:
        nv, nu, nw, _ = X.shape
        P = X.reshape(-1, 3)
        idx = np.arange(nv * nu * nw).reshape(nv, nu, nw)
        c = np.stack([idx[:-1, :-1, :-1], idx[:-1, 1:, :-1],
                      idx[1:, 1:, :-1], idx[1:, :-1, :-1],
                      idx[:-1, :-1, 1:], idx[:-1, 1:, 1:],
                      idx[1:, 1:, 1:], idx[1:, :-1, 1:]],
                     axis=-1).reshape(-1, 8)
        pts_all.append(P)
        cells_all.append(c + off)
        off += len(P)
    mesh = meshio.Mesh(np.vstack(pts_all),
                       [("hexahedron", np.vstack(cells_all))])
    mesh.write(out_path)
    print(f"wrote {out_path} ({off} points, "
          f"{sum(len(c) for c in cells_all)} hexa cells)")


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def main():
    us.set_blade_tip_corners(False)
    cs.set_emanate_outer_corners(True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== hub template (T-a) ===")
    hub = tp.run_tmesh(stl=HUB_STL, out_dir=OUT, make_plots=False, tag="hub")
    hub_result = hub["result"]
    tf_hub = hub["transform"]

    print("=== shroud geometry ===")
    shroud_mesh, tf_shroud = build_dp_data(SHROUD_STL)

    print("=== morph hub skeleton -> shroud ===")
    morph = HubToShroudMorph(hub["mesh"], shroud_mesh)
    sh_result = morph_result(hub_result, morph)
    inverted_blocks = sum(1 for b in sh_result["blocks"] if b["area"] <= 0)
    print(f"[morph] {len(sh_result['blocks'])} blocks morphed, "
          f"{inverted_blocks} inverted rings")

    # shroud grids = MORPH of the hub TFI grids: point-wise correspondence by
    # construction (independent per-surface Coons grids left the hub and
    # shroud (u,v) points laterally offset inside thin wall cells -> sheared,
    # flipped hexa cells). The morph is exact on blade + outer boundary, so
    # boundary conformity carries over from the hub grid.
    tfi_h = hub["tfi"]
    grids_s = [morph(G.reshape(-1, 2)).reshape(G.shape)
               for G in tfi_h["grids"]]
    inv_s = tot_s = 0
    for G in grids_s:
        i, t = tp._inverted_cells(G)
        inv_s += i
        tot_s += t
    # seam conformity of the morphed grids (hub seam samples through morph)
    seam_dev = None
    if tfi_h["seam_pts"]["L"] and tfi_h["seam_pts"]["R"]:
        Lp = np.unique(np.round(morph(np.vstack(tfi_h["seam_pts"]["L"])),
                                12), axis=0)
        Rp = np.unique(np.round(morph(np.vstack(tfi_h["seam_pts"]["R"])),
                                12), axis=0) \
            - np.array([float(shroud_mesh.pitch_norm), 0.0])
        d = np.linalg.norm(Lp[:, None, :] - Rp[None, :, :], axis=2)
        seam_dev = float(max(d.min(axis=1).max(), d.min(axis=0).max()))
    tfi_s = {"grids": grids_s, "inverted_cells": inv_s, "total_cells": tot_s,
             "seam_dev": seam_dev}
    print(f"[shroud-tfi] morphed hub grids: inverted cells {inv_s}/{tot_s}, "
          f"seam dev={seam_dev}")

    print("=== 3D hexa interpolation ===")
    hexa = build_hexa_blocks(tfi_h["grids"], tfi_s["grids"],
                             tf_hub, tf_shroud)
    allv = np.concatenate([hexa_cell_volumes(X).ravel() for X in hexa])
    n_cells = allv.size
    min_vol = float(np.abs(allv).min())
    jmin = np.concatenate([hexa_corner_jacobians(X)[0].ravel()
                           for X in hexa])
    jmax = np.concatenate([hexa_corner_jacobians(X)[1].ravel()
                           for X in hexa])
    maj = 1.0 if (jmax > 0).sum() >= (jmax < 0).sum() else -1.0
    n_neg = int(((jmin * maj) <= 0).sum())      # any corner Jacobian flipped

    export_vtk(hexa, OUT / "hexa_blocks.vtu")
    plot_3d(hexa, OUT / "hexa_3d.png")
    plot_morph(hub_result, sh_result, morph, shroud_mesh,
               OUT / "shroud_morphed_blocks.png")

    metrics = {
        "blocks": len(hexa),
        "hexa_cells": int(n_cells),
        "non_majority_sign_cells": n_neg,
        "min_abs_cell_volume": min_vol,
        "shroud_inverted_tfi_cells": tfi_s["inverted_cells"],
        "shroud_total_tfi_cells": tfi_s["total_cells"],
        "shroud_seam_tfi_dev": tfi_s["seam_dev"],
        "shroud_inverted_block_rings": inverted_blocks,
        "n_span": N_SPAN,
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "hexa_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[hexa] GATE inverted hexa cells: {n_neg}/{n_cells} "
          f"({'PASS' if n_neg == 0 else 'FAIL'})")
    print(f"[hexa] GATE shroud TFI inverted: "
          f"{tfi_s['inverted_cells']}/{tfi_s['total_cells']} "
          f"({'PASS' if tfi_s['inverted_cells'] == 0 else 'FAIL'})")
    print(f"[hexa] {len(hexa)} hexa blocks, {n_cells} cells, "
          f"{metrics['runtime_s']}s")
    print(f"wrote {OUT}/hexa_metrics.json")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_morph(hub_result, sh_result, morph, shroud_mesh, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(19, 8))
    for ax, res, ttl in ((axs[0], hub_result, "hub template (T-a)"),
                         (axs[1], sh_result, "morphed onto shroud")):
        for blk in res["blocks"]:
            ring = blk["ring"]
            ax.fill(ring[:, 0], ring[:, 1], alpha=0.2, color="C0")
            for s in blk["sides"]:
                ax.plot(s[:, 0], s[:, 1], "C0", lw=1.1)
        ax.set_aspect("equal")
        ax.set_title(ttl)
    # actual shroud blade + walls over the morphed panel
    bl = np.asarray(shroud_mesh.blade_loops[0], float)
    axs[1].plot(bl[:, 0], bl[:, 1], "red", lw=1.0, ls="--",
                label="actual shroud blade loop")
    xy = shroud_mesh.x[:, 0:2].numpy()
    tris = shroud_mesh.faces.T.numpy()
    axs[1].triplot(xy[:, 0], xy[:, 1], tris, lw=0.08, color="0.94", zorder=0)
    axs[1].legend(loc="upper left", fontsize=8)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_3d(hexa, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    for X in hexa:
        nv, nu, nw, _ = X.shape
        for face in (X[:, :, 0], X[:, :, -1]):        # hub + shroud faces
            for i in range(0, nv, max(1, nv // 6)):
                ax.plot(face[i, :, 0], face[i, :, 1], face[i, :, 2],
                        "0.4", lw=0.4)
            for j in range(0, nu, max(1, nu // 6)):
                ax.plot(face[:, j, 0], face[:, j, 1], face[:, j, 2],
                        "0.4", lw=0.4)
        # block edges spanwise
        for (i, j) in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            ax.plot(X[i, j, :, 0], X[i, j, :, 1], X[i, j, :, 2],
                    "C0", lw=1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("3D hexa blocks: hub (r=0.5) -> shroud (r=1.9), "
                 f"{len(hexa)} blocks")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
