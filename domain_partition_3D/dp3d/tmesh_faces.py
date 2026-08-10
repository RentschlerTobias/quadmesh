#!/usr/bin/env python3
"""T-junction-tolerant block extraction (Ansatz T).

Replaces domain_partition's QuadFaceGenerator length-4 filter: the planar-region
extraction is the same (streamline ENDPOINTS are the graph nodes, regions come
from the planar embedding), but a region is accepted as a block when it has
EXACTLY 4 *real* corners. A node where the region boundary passes straight
through (interior angle ~180 deg, i.e. a hanging T-junction on one side of a
seam or ring) is a FLAT node and does not count as a corner; a block side may
contain any number of flat nodes.

domain_partition is read-only, so the connectivity/region logic is copied here
(tolerances identical to streamlines_to_quad_faces.QuadFaceGenerator) instead of
edited upstream.
"""

import numpy as np
import networkx as nx


def build_connectivity(streamlines, tol=1e-2):
    """Copy of QuadFaceGenerator.build_connectivity (nodes = streamline
    endpoints, merge tol 1e-2), returning numpy nodes and the (a,b)->polyline
    map. Parallel curves sharing BOTH endpoints collapse to one undirected
    edge in the region graph (upstream behaviour); their count is returned so
    the caller can warn."""
    nodes = {}
    for s in streamlines:
        s = np.asarray(s, float)
        for p in (tuple(s[0]), tuple(s[-1])):
            if not any(np.allclose(p, n, atol=tol) for n in nodes):
                nodes[p] = len(nodes)
    node_list = [np.asarray(n, float) for n in nodes]

    edges, e2s = [], {}
    used_pairs = set()
    n_parallel = 0
    for s in streamlines:
        s = np.asarray(s, float)
        start = next(j for j, n in enumerate(node_list)
                     if np.allclose(s[0], n, atol=tol))
        end = next(j for j, n in enumerate(node_list)
                   if np.allclose(s[-1], n, atol=tol))
        if (start, end) in used_pairs:
            if (end, start) in used_pairs:
                n_parallel += 1          # third+ curve on the same node pair
                continue
            edges.append((end, start))
            used_pairs.add((end, start))
            e2s[(end, start)] = s[::-1]
        else:
            edges.append((start, end))
            used_pairs.add((start, end))
            e2s[(start, end)] = s
    return np.array(node_list), edges, e2s, n_parallel


def extract_regions(edges):
    """Bounded+unbounded faces of the planar embedding (same traversal as
    QuadFaceGenerator.get_quad_faces). Returns list of node cycles, or None if
    the graph is non-planar. Self-loop edges (closed rings that survived
    splitting) are skipped."""
    graph = nx.Graph()
    graph.add_edges_from((a, b) for a, b in edges if a != b)
    if graph.number_of_edges() == 0:
        return []
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        return None
    faces_raw, seen = [], set()
    for u, v in embedding.edges():
        if (u, v) in seen:
            continue
        face = embedding.traverse_face(u, v, mark_half_edges=seen)
        faces_raw.append(face)
    return faces_raw


def edge_polyline(a, b, e2s):
    """Polyline of graph edge (a,b), oriented a -> b."""
    if (a, b) in e2s:
        return np.asarray(e2s[(a, b)], float)
    return np.asarray(e2s[(b, a)], float)[::-1]


def _dir_out(poly, arc=0.02):
    """Unit direction leaving poly[0], measured to the point at arclength
    ~`arc` along the curve (stable against a noisy first segment)."""
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    tot = float(seg.sum())
    if tot < 1e-12:
        return None
    a = min(arc, tot / 3.0) if tot / 3.0 > 1e-9 else tot
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.searchsorted(cum, a, side="right")) - 1
    i = min(i, len(poly) - 2)
    t = (a - cum[i]) / max(cum[i + 1] - cum[i], 1e-30)
    pt = poly[i] + t * (poly[i + 1] - poly[i])
    v = pt - poly[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def _ring_of(face, e2s):
    """Concatenated boundary polyline of a region (closed, last pt == first)."""
    pts = []
    n = len(face)
    for k in range(n):
        poly = edge_polyline(face[k], face[(k + 1) % n], e2s)
        pts.append(poly[:-1])
    pts.append(np.asarray([edge_polyline(face[-1], face[0], e2s)[-1]]))
    return np.vstack(pts)


def _dist_to_polyline(p, poly):
    """Distance from point p to a polyline (min over segments)."""
    p = np.asarray(p, float)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(L2, 1e-30),
                0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p, axis=1).min())


def _is_wall_segment(poly, boundary_dist_fn, bnd_tol=1e-3):
    """True if every point of the polyline lies within bnd_tol of the boundary."""
    poly = np.asarray(poly, float)
    return all(boundary_dist_fn(p) < bnd_tol for p in poly)


def _edge_is_wall(a, b, e2s, boundary_dist_fn, bnd_tol=1e-3):
    """True if the whole polyline of edge (a,b) is a wall segment."""
    return _is_wall_segment(edge_polyline(a, b, e2s), boundary_dist_fn, bnd_tol)


def _shoelace(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def corner_turns(face, e2s):
    """Turn angle (deg) of the region boundary at every cycle node.
    turn = 180 - interior angle; ~0 at a flat (hanging T) node."""
    n = len(face)
    turns = []
    for k in range(n):
        prv, cur, nxt = face[k - 1], face[k], face[(k + 1) % n]
        din = _dir_out(edge_polyline(cur, prv, e2s))   # towards prv
        dout = _dir_out(edge_polyline(cur, nxt, e2s))  # towards nxt
        if din is None or dout is None:
            turns.append(180.0)
            continue
        arrival = -din                                  # direction of travel at cur
        cosang = float(np.clip(np.dot(arrival, dout), -1.0, 1.0))
        turns.append(float(np.degrees(np.arccos(cosang))))
    return turns


def classify_corners(face, e2s, flat_tol_deg=25.0, boundary_dist_fn=None,
                     bnd_tol=1e-3, min_turn_deg=0.5):
    """Indices (into the cycle) of the REAL corners of a region: nodes where
    the boundary turns by more than flat_tol_deg. |180 - alpha| < flat_tol
    (alpha = interior angle) <=> turn angle < flat_tol => FLAT (T-node).
    When boundary_dist_fn is provided, a node with exactly one adjacent wall
    segment and at least one separatrix is also promoted to a corner if its
    turn exceeds min_turn_deg."""
    turns = corner_turns(face, e2s)
    corners = []
    n = len(face)
    for k, t in enumerate(turns):
        if t > flat_tol_deg:
            corners.append(k)
            continue
        if boundary_dist_fn is None or t <= min_turn_deg:
            continue
        prv, cur, nxt = face[k - 1], face[k], face[(k + 1) % n]
        wa = _edge_is_wall(cur, prv, e2s, boundary_dist_fn, bnd_tol)
        wb = _edge_is_wall(cur, nxt, e2s, boundary_dist_fn, bnd_tol)
        if wa != wb:  # exactly one wall, at least one separatrix
            corners.append(k)
    return corners


class TMeshFaceGenerator:
    """Region -> block extraction tolerating hanging T-nodes.

    get_blocks() returns a dict:
      nodes      (N,2) graph nodes (streamline endpoints)
      blocks     list of dicts: cycle (node ids, CCW), corner_pos (4 indices
                 into cycle), corners (4 node ids), sides (4 polylines, side k
                 runs corner k -> corner k+1), ring (closed outline)
      rejects    regions with != 4 real corners (dicts w/ cycle, n_real, ring)
      outer_idx / blade_idx: region indices dropped as unbounded face / holes
      planar     False when the graph was non-planar (no blocks then)
      n_parallel collapsed parallel edges (upstream limitation)
    """

    def __init__(self, streamlines, blade_loops=None, flat_tol_deg=25.0,
                 verbose=True, boundary_dist_fn=None, bnd_tol=1e-3):
        self.streamlines = list(streamlines)
        self.blade_loops = blade_loops or []
        self.flat_tol_deg = float(flat_tol_deg)
        self.verbose = verbose
        self.boundary_dist_fn = boundary_dist_fn
        self.bnd_tol = float(bnd_tol)

    def get_blocks(self):
        nodes, edges, e2s, n_parallel = build_connectivity(self.streamlines)
        if n_parallel and self.verbose:
            print(f"[tmesh] WARNING: {n_parallel} parallel curves collapsed "
                  f"(regions between them are lost)")
        regions = extract_regions(edges)
        out = {"nodes": nodes, "edges": edges, "e2s": e2s,
               "n_parallel": n_parallel, "blocks": [], "rejects": [],
               "planar": regions is not None, "outer_idx": [],
               "blade_idx": [], "blade_regions": []}
        if regions is None:
            if self.verbose:
                print("[tmesh] region graph NON-PLANAR -> no blocks")
            return out

        infos = []
        for face in regions:
            if len(face) < 3 or len(set(face)) != len(face):
                infos.append({"cycle": list(face), "ring": None, "area": 0.0})
                continue
            ring = _ring_of(face, e2s)
            area = _shoelace(ring)
            if area < 0:                       # normalize to CCW
                face = list(face)[::-1]
                ring = _ring_of(face, e2s)
                area = _shoelace(ring)
            infos.append({"cycle": list(face), "ring": ring, "area": area})

        # unbounded outer face: largest |area| of the cycle outline
        areas = [abs(i["area"]) for i in infos]
        outer = int(np.argmax(areas)) if infos else -1
        loops = [np.asarray(bl, float) for bl in self.blade_loops]

        def _on_blade(ring, tol=0.005):
            """Blade HOLE = the region whose entire outline lies on a blade
            loop. A centroid-inside test is wrong here: O-grid regions wrapping
            the blade have their outline centroid inside the blade too."""
            sub = ring[:: max(1, len(ring) // 24)]
            for bl in loops:
                if all(_dist_to_polyline(p, bl) < tol for p in sub):
                    return True
            return False

        for ri, info in enumerate(infos):
            if ri == outer:
                out["outer_idx"].append(ri)
                continue
            if info["ring"] is None:           # degenerate walk (bridge/stub)
                out["rejects"].append({**info, "n_real": -1})
                continue
            centroid = info["ring"][:-1].mean(axis=0)
            if _on_blade(info["ring"]):
                out["blade_idx"].append(ri)
                out["blade_regions"].append(info)
                continue
            corner_pos = classify_corners(
                info["cycle"], e2s, self.flat_tol_deg,
                boundary_dist_fn=self.boundary_dist_fn,
                bnd_tol=self.bnd_tol)
            if len(corner_pos) != 4:
                out["rejects"].append({**info, "n_real": len(corner_pos),
                                       "centroid": centroid})
                continue
            cyc = info["cycle"]
            n = len(cyc)
            sides, side_chains = [], []
            for k in range(4):
                a, b = corner_pos[k], corner_pos[(k + 1) % 4]
                chain = [cyc[(a + j) % n] for j in range((b - a) % n + 1)]
                side = []
                for j in range(len(chain) - 1):
                    poly = edge_polyline(chain[j], chain[j + 1], e2s)
                    side.append(poly[:-1] if j < len(chain) - 2 else poly)
                sides.append(np.vstack(side))
                side_chains.append(chain)
            out["blocks"].append({
                "cycle": cyc, "ring": info["ring"], "area": info["area"],
                "corner_pos": corner_pos,
                "corners": [cyc[p] for p in corner_pos],
                "sides": sides, "side_chains": side_chains,
                "centroid": centroid})

        if self.verbose:
            print(f"[tmesh] {len(regions)} regions -> {len(out['blocks'])} "
                  f"blocks, {len(out['rejects'])} rejected (!=4 real corners), "
                  f"{len(out['blade_idx'])} blade holes, outer dropped")
        return out


def node_regularity(result, boundary_dist_fn, bnd_tol=1e-5,
                    flat_tol_deg=15.0, classify_bnd_tol=1e-3):
    """Interior-node classification on real-corner incidence:
      regular    real corner in 4 regions, flat in 0
      t-node     real corner in 2 regions, flat in 1 (hanging junction)
      irregular  anything else (e.g. field-singularity nodes, defects)
    boundary_dist_fn(p) -> distance of p to the domain boundary (walls+blade);
    boundary nodes are excluded like in partition_surface._block_annotations."""
    nodes = result["nodes"]
    real_cnt = np.zeros(len(nodes), int)
    flat_cnt = np.zeros(len(nodes), int)
    e2s = result["e2s"]
    regs = result["blocks"] + result["rejects"]
    for reg in regs:
        cyc = reg.get("cycle")
        if not cyc or reg.get("ring") is None:
            continue
        rp = set(reg.get("corner_pos") if reg.get("corner_pos") is not None
                 else classify_corners(cyc, e2s, flat_tol_deg,
                                       boundary_dist_fn=boundary_dist_fn,
                                       bnd_tol=classify_bnd_tol))
        for k, nid in enumerate(cyc):
            if k in rp:
                real_cnt[nid] += 1
            else:
                flat_cnt[nid] += 1
    regular, tnodes, irregular = [], [], []
    for i, p in enumerate(nodes):
        if boundary_dist_fn(p) < bnd_tol:
            continue
        r, f = int(real_cnt[i]), int(flat_cnt[i])
        if r == 0 and f == 0:
            continue
        if r == 4 and f == 0:
            regular.append(i)
        elif r == 2 and f == 1:
            tnodes.append(i)
        else:
            irregular.append(i)
    return regular, tnodes, irregular
