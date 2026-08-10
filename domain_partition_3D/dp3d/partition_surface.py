#!/usr/bin/env python3
"""
Stage 1 / Step C: Run domain_partition's 2D cross-field block partition on the
unwrapped surface mesh.

Pipeline (mirrors domain_partition/data_generator.py:get_mesh):
    FrameField -> detect_singularities -> StreamlineGenerator_v2
               -> StreamlinePostProcessor -> block_mesh

One deviation: StreamlineGenerator_v2 hardcodes the v1 ``SeparatrixGenerator``,
whose vector interpolation divides by a zero-norm field near singularities and
returns ``None`` (the NACA data-gen never hit it because it filters to
near-zero-singularity cases). We monkeypatch in ``SeparatrixGenerator_v2``,
which guards that case. No edits to the domain_partition repo.

Returns the 2D block mesh (block_mesh.x = (N,2) corners, block_mesh.faces =
(4,B) quad blocks) plus the affine (s,t)<-[0,1]^2 transform for back-mapping.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from torch_geometric.data import Data

# --- swap in our robust separatrix emanation finder ---
# the vendored v1/v2 generators are buggy and untested on real
# singularities; only the RK/Heun integrator in StreamlineGenerator_v2 is sound.
from .field import streamline_generator_v2 as _sg2
from .clean_separatrix import CleanSeparatrixGenerator
_sg2.SeparatrixGenerator = CleanSeparatrixGenerator

from .field import FrameField, StreamlineGenerator_v2, StreamlinePostProcessor
from .field.singularity_detector import detect_singularities
from .field.quad_partition_validator import QuadPartitionValidator
from .field.streamline_merging import StreamlineMerging
from .dp_adapter import build_dp_data


# --- enable Xiao 2020 Algorithm 2 Case 2 in StreamlineMerging -----------------
# StreamlineMerging.find_missed_streamline_endpoints implements Case 2 (Fig 8b):
# a streamline that PASSES a singularity (angle ~180 deg to a reference out-going
# streamline, within 45 deg) but does not terminate there is CUT onto it. The
# upstream __init__ defines it but never calls it, so wrap separatrices that
# graze the opposite blade tip never connect and the O-grid ring never closes.
# We reinstate the call between prepare and merge (no edit to domain_partition).
def _streamline_merging_init(self, mesh, verbose=True):
    self.verbose = verbose
    self.Streamlines, self.Singularities = self.prepare_streamlines(mesh)
    self.Streamlines, self.Singularities = \
        self.find_missed_streamline_endpoints(self.Streamlines, self.Singularities)
    self.new_streamlines = self.merge_streamlines(
        self.Streamlines, self.Singularities,
        getattr(mesh, "expected_separatrices", None), mesh)


StreamlineMerging.__init__ = _streamline_merging_init


# A streamline must keep integrating across the periodic theta seam until it
# terminates on a NON-periodic boundary (inlet/outlet/blade) or a termination
# node -- however many pitches that takes. MAX_WRAPS only guards against a
# closed circular orbit that would never terminate (infinite loop); it must be
# large enough for a helical spiral to CONVERGE onto its limit orbit before the
# cap (the convergence-based merge needs a few windings of room).
MAX_WRAPS = 8


def _wrap_shifts(s_coord, pitchn):
    """Candidate s-shifts k*pitch (k integer) that map a universal-cover point
    back into the mesh's normalized s-range, nearest wrap counts first. Returns
    [0.0] when periodicity is off; [] when the point exceeds MAX_WRAPS."""
    if not (TILE_PERIODIC and pitchn):
        return [0.0]
    k_c = int(round((0.5 - s_coord) / pitchn))
    if abs(k_c) > MAX_WRAPS:
        return []
    ks = {0, k_c - 1, k_c, k_c + 1}
    return [k * pitchn for k in sorted(ks, key=abs) if abs(k) <= MAX_WRAPS]


def _robust_find_containing_face(self, point, mesh):
    """Robust point location. The upstream version only tests faces incident to
    the single nearest node, so a streamline stepping into a non-incident face
    drops out of the mesh mid-domain and the separatrix dies early (-> dangling
    stubs that never reach their target). We test faces of the k nearest nodes,
    then fall back to a brute-force scan.

    Periodic wrap (Phase 2b): when the point leaves the mesh through a theta
    seam, some +/- k-pitch image still lies inside -- the streamline physically
    continues in a neighbour passage and must keep integrating until it reaches
    a NON-periodic boundary or termination node. We retry the lookup at every
    candidate wrap image (universal-cover coordinates), capped at MAX_WRAPS
    against closed circular orbits. Returns a face index or None."""
    p = point.detach().numpy() if hasattr(point, "detach") else np.asarray(point)
    nodes = mesh.x[:, 0:2].numpy()
    faces = mesh.faces.numpy()

    def _locate(q):
        d = np.sum((nodes - q) ** 2, axis=1)
        near = np.argsort(d)[:6]
        cand = []
        for nid in near:
            cand.extend(mesh.nodes_faces_ids.get(int(nid), []))
        for fi in dict.fromkeys(cand):
            if _point_in_tri(q, nodes[faces[:, fi]]):
                return fi
        for fi in range(faces.shape[1]):  # brute fallback
            if _point_in_tri(q, nodes[faces[:, fi]]):
                return fi
        return None

    for sh in _wrap_shifts(float(p[0]), getattr(mesh, "pitch_norm", None)):
        fi = _locate(p + np.array([sh, 0.0]) if sh else p)
        if fi is not None:
            return fi
    return None


def _point_in_tri(p, tri, eps=1e-9):
    v0 = tri[2] - tri[0]
    v1 = tri[1] - tri[0]
    v2 = p - tri[0]
    d00 = v0 @ v0; d01 = v0 @ v1; d02 = v0 @ v2
    d11 = v1 @ v1; d12 = v1 @ v2
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return False
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1 + eps)


StreamlineGenerator_v2.find_containing_face = _robust_find_containing_face


def _periodic_get_best_cross_vector(self, point, previous_direction, mesh,
                                    containing_face_idx):
    """StreamlineGenerator_v2.get_best_cross_vector with periodic wrap: when the
    containing face was located for the point's +/- one-pitch image (cover point
    left the mesh through a theta seam), the barycentric interpolation must use
    that wrapped image, not the raw cover point. We pick the candidate in
    {p, p-pitch, p+pitch} whose barycentric coordinates lie inside the face.
    Cross-arm construction and best-dot selection are verbatim upstream (the
    returned direction is translation-invariant)."""
    if containing_face_idx is None:
        print(f'point ({point}) is not inside a face')
        return None, mesh

    face_indices = mesh.faces[:, containing_face_idx]
    vertices = mesh.x[face_indices, 0:2]
    ref_vecs = (mesh.u[face_indices]).to(torch.float)

    pitchn = getattr(mesh, "pitch_norm", None)
    cands = [point + torch.tensor([sh, 0.0], dtype=point.dtype) if sh else point
             for sh in _wrap_shifts(float(point[0]), pitchn)]
    if not cands:
        return None, mesh
    bary_coords, best_min = None, -torch.inf
    for q in cands:
        try:
            bc = self.compute_barycentric_coordinates(q, vertices)
        except ValueError:
            continue
        mn = float(bc.min())
        if mn > best_min:
            best_min, bary_coords = mn, bc
        if mn >= -1e-9:
            break  # inside; earlier candidates take precedence
    if bary_coords is None:
        return None, mesh

    interpolated_vec = torch.einsum('i,ij->j', bary_coords, ref_vecs)
    n = torch.norm(interpolated_vec)
    if n < 1e-12:
        return None, mesh
    interpolated_vec = interpolated_vec / n

    base_angle = torch.atan2(interpolated_vec[1], interpolated_vec[0]) / 4
    max_dot, best_vector = -float('inf'), None
    for i in range(4):
        angle = base_angle + i * (torch.pi / 2)
        v = torch.tensor([torch.cos(angle), torch.sin(angle)])
        dot = torch.dot(previous_direction, v)
        if dot > max_dot:
            max_dot, best_vector = dot, v
    return best_vector, mesh


StreamlineGenerator_v2.get_best_cross_vector = _periodic_get_best_cross_vector


def _periodic_check_termination(self, point, face_idx, origin):
    """StreamlineGenerator_v2.check_termination_criteria with periodic wrap: a
    streamline continuing in the neighbour passage (cover coords) must still
    terminate on the neighbour's singularities/corners. We test the point's
    {0, -pitch, +pitch} images against the central termination nodes and shift a
    hit BACK into cover coordinates so the recorded polyline stays continuous."""
    if face_idx is None:
        return True, point
    pitchn = getattr(self.mesh, "pitch_norm", None)
    for sh in _wrap_shifts(float(point[0]), pitchn):
        shv = torch.tensor([sh, 0.0], dtype=point.dtype)
        pw = point + shv if sh else point
        distances = torch.linalg.norm(self.streamline_termination_nodes - pw,
                                      dim=1)
        min_distance = torch.min(distances)
        if min_distance < self.termination_node_range:
            idx = torch.where(distances == min_distance)[0]
            tn = self.streamline_termination_nodes[idx.item(), :]
            if torch.linalg.norm(point - origin) < self.termination_node_range:
                return False, None  # directly at the start
            return True, (tn - shv if sh else tn)
    return False, None


StreamlineGenerator_v2.check_termination_criteria = _periodic_check_termination


def _add_cross_at_boundaries_fixed(self):
    """Corrected FrameField.add_cross_at_boundaries.

    The upstream version stores the boundary Dirichlet cross at array position
    ``i`` (the loop counter over boundary nodes), but the solver in
    ``compute_initial_frame_field`` reads ``frame_field_coords[node_id]``. These
    only agree when boundary node ids happen to be 0..K-1 (true for the NACA
    gmsh meshes, false for a welded STL with scattered ids) -> scrambled BC ->
    spurious singularities. We store at the node id instead. Geometry/angle
    logic is verbatim from the upstream method.
    """
    pi = torch.pi
    mesh = self.mesh
    num_nodes = mesh.x.size(0)
    mask_boundaryEdges = mesh.edge_attr == 1
    idx_boundaryNodes = torch.unique(mesh.edge_index[0, mask_boundaryEdges])
    boundary_edges = mesh.edge_index[:, mask_boundaryEdges]

    frame_field_angle = torch.zeros((num_nodes), dtype=torch.float)
    frame_field_coords = torch.zeros((num_nodes, 2), dtype=torch.float)

    for idx_current_node in idx_boundaryNodes:
        nid = int(idx_current_node)
        boundary_edges_of_node = torch.where(boundary_edges[0, :] == idx_current_node)[0]
        neighbours_idx = boundary_edges[1, boundary_edges_of_node]
        source_node = mesh.x[idx_current_node, 0:2]
        destination_node0 = mesh.x[neighbours_idx[0], 0:2]
        destination_node1 = mesh.x[neighbours_idx[1], 0:2]

        edge0 = destination_node0 - source_node
        edge1 = destination_node1 - source_node
        edge0_normalized = edge0 / torch.norm(edge0, p=2)
        edge1_normalized = edge1 / torch.norm(edge1, p=2)

        angle0 = torch.atan2(edge0_normalized[1], edge0_normalized[0]) % (2 * pi)
        angle1 = torch.atan2(edge1_normalized[1], edge1_normalized[0]) % (2 * pi)

        # Average in the REPRESENTATIVE (4-theta) space, not in tangent space.
        # The upstream code averages the two wall tangents (edge0+edge1) and only
        # THEN maps to the cross representative. At a slanted (non-90deg) corner
        # that tangent-space mean disagrees with both adjacent walls and forces a
        # spurious singularity right next to each c0 corner. The cross field lives
        # in representative space, so the mean must be taken there: map each wall
        # tangent to its representative, then circular-mean them. For smooth
        # boundary nodes the two edges are ~collinear and both methods coincide.
        r0 = self.map_cross_vectors_to_reference_vector(angle0)
        r1 = self.map_cross_vectors_to_reference_vector(angle1)
        ref_angle = torch.atan2(torch.sin(r0) + torch.sin(r1),
                                torch.cos(r0) + torch.cos(r1))

        frame_field_angle[nid] = ref_angle
        frame_field_coords[nid, 0] = torch.cos(ref_angle)
        frame_field_coords[nid, 1] = torch.sin(ref_angle)

    self.mesh.frame_field_angle = frame_field_angle
    self.mesh.frame_field_coords = frame_field_coords


FrameField.add_cross_at_boundaries = _add_cross_at_boundaries_fixed


# --- optional soft boundary conditions (the "energy" knob) -----------------
# BC_WEIGHT = None reproduces the upstream HARD Dirichlet BC (row-replacement).
# A finite w adds a penalty term  w*|u - u_wall|^2  to the harmonic energy
# instead of hard-fixing the boundary DOFs: the harmonic smoothing then governs
# the interior more strongly and tends to merge/annihilate nearby opposite cross
# singularities. Lower w = softer boundary = smoother interior (but weaker wall
# alignment). Note generate_cross_field re-imposes the exact BC into
# mesh.frame_field; mesh.u (used downstream for singularities/separatrices) keeps
# the soft solve, which is exactly the relaxed interior we want.
BC_WEIGHT = None

# --- pitchwise periodicity (Phase 2) ---------------------------------------
# When True and mesh.periodic_pairs is non-empty, the two theta walls are NOT
# wall-aligned Dirichlet boundaries; instead the seam is glued: each slave node
# DOF is identified with its master partner (u_slave = u_master) and the slave's
# triangle stiffness is welded onto the master, giving a true periodic Laplacian.
# This removes the artificial wall holonomy on the pitch boundaries (Kowalski Eq
# 14 budget) that forces extra interior singularities. Kept behind a flag so the
# wall-BC result stays reproducible for comparison.
PERIODIC = True


def set_bc_weight(w):
    global BC_WEIGHT
    BC_WEIGHT = w


def set_periodic(flag):
    global PERIODIC
    PERIODIC = bool(flag)


# --- periodic block tiling (Phase 2) ---------------------------------------
# The theta-seam is periodic in the field solve, but the block graph
# (QuadFaceGenerator) uses streamline endpoints as nodes: the left-seam and
# right-seam nodes are distinct (offset by one pitch in s), so block regions do
# not close across the seam and leave irregular interior nodes. When True (and
# mesh.pitch_norm is set) we replicate every streamline by +/- one pitch in s to
# a 3-pitch strip, build blocks on the flat strip (still planar), then keep only
# the faces whose centroid lies in the central pitch (the physical passage). This
# closes the seam and needs no domain_partition edit.
TILE_PERIODIC = True


def set_tile_periodic(flag):
    global TILE_PERIODIC
    TILE_PERIODIC = bool(flag)


def set_periodicity(flag):
    """Master switch for ALL pitchwise periodicity: field seam weld (PERIODIC)
    AND block-stage wrap integration/tiling/helix merge (TILE_PERIODIC).
    set_periodicity(False) treats the theta walls as plain walls everywhere."""
    set_periodic(flag)
    set_tile_periodic(flag)


def _tile_streamlines_periodic(streamlines, pitch_norm):
    """Replicate every streamline by whole pitches in normalized s (t unchanged)
    so that every copy overlapping the widened strip [-pitch, 1+pitch] exists.
    Curves may span several pitches (they integrate across the seam until they
    hit a non-periodic boundary), so the shift range is per curve, derived from
    its s-extent -- not a fixed +/-1.

    Copies that coincide with an existing curve are dropped: the left seam wall
    is the exact -pitch translate of the right one, so tiling would lay duplicate
    polylines on top of each other -- coincident curves make the intersection
    splitter pathological and clutter the block graph. Key = sorted endpoints +
    centroid, rounded to the seam-conformity tolerance (1e-3)."""
    def _key(s):
        a, b, m = s[0], s[-1], s.mean(axis=0)
        if (a[0], a[1]) > (b[0], b[1]):
            a, b = b, a
        return tuple(np.round(np.concatenate([a, b, m]), 3))

    lo, hi = -pitch_norm, 1.0 + pitch_norm      # widened strip to cover
    orig = [np.asarray(s, float) for s in streamlines]
    tiled = list(orig)
    seen = {_key(s) for s in orig}
    n_dup = 0
    for s in orig:
        smin, smax = s[:, 0].min(), s[:, 0].max()
        k_lo = int(np.ceil((lo - smax) / pitch_norm))
        k_hi = int(np.floor((hi - smin) / pitch_norm))
        for k in range(k_lo, k_hi + 1):
            if k == 0:
                continue
            c = s.copy()
            c[:, 0] = c[:, 0] + k * pitch_norm
            key = _key(c)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            tiled.append(c)
    if n_dup:
        print(f"[tile] deduped {n_dup} coincident copies (seam walls)")
    return tiled


def _central_parallelogram(mesh):
    """matplotlib Path of the central passage = the sheared parallelogram spanned
    by the four OUTER (corner_type==0) domain corners, ordered around the loop
    (bottom edge left->right, top edge right->left). Used to select the central
    pitch after tiling; robust to the shear (not an axis-aligned band)."""
    from matplotlib.path import Path as _MplPath
    ct = mesh.corner_type.numpy() if hasattr(mesh.corner_type, "numpy") \
        else np.asarray(mesh.corner_type)
    corners = mesh.x[:, 0:2].numpy()[ct == 0]
    if len(corners) < 3:
        return None
    tmid = 0.5 * (corners[:, 1].min() + corners[:, 1].max())
    bottom = corners[corners[:, 1] <= tmid]
    top = corners[corners[:, 1] > tmid]
    bottom = bottom[np.argsort(bottom[:, 0])]           # left -> right
    top = top[np.argsort(-top[:, 0])]                   # right -> left
    ring = np.vstack([bottom, top])
    return _MplPath(ring)


def _extract_central_blocks(block_mesh, central_path):
    """Keep only quad faces whose centroid lies in the central parallelogram, then
    renumber the surviving corner nodes. Rebuilds x/faces/edge_index so both the
    validator and the plots work on the trimmed block mesh."""
    if block_mesh.faces is None or block_mesh.faces.numel() == 0:
        return block_mesh
    bx = block_mesh.x.numpy()
    bf = block_mesh.faces.numpy().T                     # (B,4)
    cent = bx[bf].mean(axis=1)                          # (B,2)
    keep = central_path.contains_points(cent)
    kept = bf[keep]
    if len(kept) == 0:
        return block_mesh
    used = np.unique(kept)
    remap = {int(o): i for i, o in enumerate(used)}
    new_x = torch.tensor(bx[used], dtype=block_mesh.x.dtype)
    new_faces = torch.tensor(
        np.vectorize(remap.get)(kept).T, dtype=torch.long)   # (4,B')
    # undirected quad edges for the validator
    e = []
    for q in new_faces.T.tolist():
        for k in range(4):
            e.append((q[k], q[(k + 1) % 4]))
    edge_index = torch.tensor(e, dtype=torch.long).T if e else \
        torch.zeros((2, 0), dtype=torch.long)
    return Data(x=new_x, faces=new_faces, edge_index=edge_index)


def _seam_weld_map(mesh, num_nodes):
    """Return (weld, seam, slave_of) for the periodic seam.

    weld: (num_nodes,) int array remapping each slave node id to its master.
    seam: set of all node ids on the seam (masters + slaves).
    slave_of: {slave_id: master_id}."""
    weld = np.arange(num_nodes)
    seam, slave_of = set(), {}
    pairs = getattr(mesh, "periodic_pairs", None)
    if not PERIODIC or pairs is None or len(pairs) == 0:
        return weld, seam, slave_of
    pp = pairs.numpy() if hasattr(pairs, "numpy") else np.asarray(pairs)
    for m, s in pp:
        m, s = int(m), int(s)
        weld[s] = m
        slave_of[s] = m
        seam.add(m); seam.add(s)
    return weld, seam, slave_of


def _compute_initial_frame_field_soft(self):
    """FrameField.compute_initial_frame_field with optional soft (penalty) BC and
    optional pitchwise periodicity.

    BC_WEIGHT: None -> hard Dirichlet wall BC, finite -> soft penalty BC.
    PERIODIC: glue the theta-seam (weld slave stiffness onto master + constrain
    u_slave = u_master), and skip wall-Dirichlet on seam nodes."""
    num_nodes = self.mesh.x.shape[0]
    num_elements = self.mesh.faces.shape[1]
    nodes = self.mesh.x[:, 0:2]
    elements = self.mesh.faces.T
    mask_boundaryEdges = self.mesh.edge_attr == 1
    boundary_nodes_indices = torch.unique(self.mesh.edge_index[0, mask_boundaryEdges])
    num_dofs = num_nodes * 2
    b = np.zeros(num_dofs)

    weld, seam, slave_of = _seam_weld_map(self.mesh, num_nodes)
    self._seam_nodes = seam

    rows, cols, vals = [], [], []
    for e in range(num_elements):
        nodes_indices = elements[e]
        coords = nodes[nodes_indices]
        A_e = self.compute_local_stiffness_matrix(coords)
        dof_indices = np.empty(6, dtype=int)
        for i in range(3):
            wn = int(weld[int(nodes_indices[i])])  # periodic remap (identity if none)
            dof_indices[2 * i] = 2 * wn
            dof_indices[2 * i + 1] = 2 * wn + 1
        for i_local in range(6):
            for j_local in range(6):
                rows.append(dof_indices[i_local])
                cols.append(dof_indices[j_local])
                vals.append(A_e[i_local, j_local])

    A = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr().tolil()
    for idx in boundary_nodes_indices:
        idx = int(idx)
        if idx in seam:           # periodic seam: no wall Dirichlet
            continue
        dof_x, dof_y = 2 * idx, 2 * idx + 1
        bx = float(self.mesh.frame_field_coords[idx, 0])
        by = float(self.mesh.frame_field_coords[idx, 1])
        if BC_WEIGHT is None:
            A.rows[dof_x] = [dof_x]; A.data[dof_x] = [1.0]
            A.rows[dof_y] = [dof_y]; A.data[dof_y] = [1.0]
            b[dof_x] = bx; b[dof_y] = by
        else:
            w = float(BC_WEIGHT)
            A[dof_x, dof_x] += w
            A[dof_y, dof_y] += w
            b[dof_x] += w * bx
            b[dof_y] += w * by

    # periodic constraint rows: u_slave - u_master = 0 (slave stiffness already
    # welded onto master above, so the master row carries the periodic stencil).
    for s, m in slave_of.items():
        for comp in (0, 1):
            ds, dm = 2 * s + comp, 2 * m + comp
            A.rows[ds] = [ds, dm]
            A.data[ds] = [1.0, -1.0]
            b[ds] = 0.0

    A_sparse = A.tocsr()
    u = spsolve(A_sparse, b)
    return A_sparse, b, u


FrameField.compute_initial_frame_field = _compute_initial_frame_field_soft


def _generate_cross_field_seam_aware(self):
    """FrameField.generate_cross_field, but the boundary re-imposition skips the
    periodic seam nodes. The upstream method overwrites every boundary node in
    mesh.frame_field with its wall-aligned Dirichlet cross; on the periodic seam
    that would reintroduce the wall holonomy into singularity detection. We keep
    the solved periodic value on the seam instead."""
    A, b, u = self.compute_initial_frame_field()
    u_new = self.Linearization_Norm_Constraint(A, b, u)

    u_init = torch.cat((torch.from_numpy(u[::2]).unsqueeze(1),
                        torch.from_numpy(u[1::2]).unsqueeze(1)), dim=1)
    u_final = torch.cat((torch.from_numpy(u_new[::2]).unsqueeze(1),
                         torch.from_numpy(u_new[1::2]).unsqueeze(1)), dim=1)

    seam = getattr(self, "_seam_nodes", set())
    vector_field = u_final.detach().clone().to(torch.float)
    bidx = torch.where(self.mesh.x[:, 2] != 2)[0].tolist()
    for i in bidx:
        if i in seam:             # periodic seam keeps the solved value
            continue
        vector_field[i, :] = self.mesh.frame_field_coords[i, :]
    self.mesh.u_init = u_init
    self.mesh.u = u_final
    self.mesh.frame_field = vector_field


FrameField.generate_cross_field = _generate_cross_field_seam_aware


def partition(stl_path, verbose=True, bc_weight=None):
    """Run cross-field block partition on the unwrapped surface.

    bc_weight: None = hard Dirichlet boundary (default); a finite float = soft
    penalty BC (the energy knob, see set_bc_weight) that relaxes the interior to
    merge spurious cross singularities.

    Returns (block_mesh, mesh, transform):
      block_mesh.x      (Nc,2) block corner nodes in normalized [0,1]^2
      block_mesh.faces  (4,B)  quad block connectivity
      mesh              the triangulated mesh w/ field, separatrices, streamlines
      transform         affine map normalized [0,1]^2 -> (s,t)
    """
    set_bc_weight(bc_weight)
    mesh, transform = build_dp_data(stl_path)

    ff = FrameField(mesh)
    m = detect_singularities(ff.mesh)
    n_sing = int((m.singularities != 0).sum())

    sl = StreamlineGenerator_v2(ff.mesh)
    _drop_degenerate_corner_seps(sl.mesh)
    _close_helical_streamlines(sl.mesh)
    _emit_dock_crossings(sl)
    _snap_separatrix_endpoints(sl.mesh, radius=0.045)

    # periodic tiling: replicate streamlines +/- one pitch, build blocks on the
    # 3-pitch strip, keep the central passage (closes the theta-seam).
    central_path = None
    pitch_norm = getattr(sl.mesh, "pitch_norm", None)
    if TILE_PERIODIC and pitch_norm:
        central_path = _central_parallelogram(sl.mesh)
        sl.mesh.streamlines = _tile_streamlines_periodic(
            sl.mesh.streamlines, pitch_norm)
        print(f"[tile] periodic strip: pitch_norm={pitch_norm:.4f}, "
              f"{len(sl.mesh.streamlines)} streamlines (x3)")

    pp = StreamlinePostProcessor(sl.mesh, verbose=False)
    block_mesh = pp.block_mesh
    if central_path is not None:
        n_all = block_mesh.faces.shape[1] if block_mesh.faces is not None else 0
        block_mesh = _extract_central_blocks(block_mesh, central_path)
        n_ctr = block_mesh.faces.shape[1] if block_mesh.faces is not None else 0
        print(f"[tile] central extraction: {n_all} strip blocks -> {n_ctr} "
              f"central blocks")
    n_blocks = block_mesh.faces.shape[1] if block_mesh.faces is not None else 0
    if verbose:
        print(f"[partition] singularities={n_sing}  "
              f"separatrices={len(sl.mesh.separatrices)}  "
              f"-> {n_blocks} quad blocks ({block_mesh.x.shape[0]} corners)")
    return block_mesh, sl.mesh, transform


def _close_helical_streamlines(mesh, tol=0.015, align=0.9, min_wraps=2.0,
                               ring_dup_tol=0.06):
    """Xiao 2020 circular-streamline closing, periodic (mod-pitch) variant with
    a CONVERGENCE criterion.

    On the periodic band the LE/TE prongs heading for inlet/outlet never reach a
    non-periodic boundary: they spiral helically around the wheel, converging to
    a closed limit orbit. The spiral must NOT be closed at its first re-entry
    (winding gaps are still ~0.03-0.05 there; closing that early breaks the
    partition): it is merged onto the orbit only once it is genuinely close to
    its own previous winding AND has already made a couple of laps:
      - tight proximity  |P_j - (P_i + k*pitch)| < tol (~0.015),
      - same direction   tangent dot > align (0.9),
      - min laps         net s-drift before the hit >= min_wraps * pitch.

    A helical curve then becomes TWO objects (analogous to the blade O-ring +
    dock): (1) the finite prong P[:i+1] -- keeps ALL earlier spiral windings and
    is later docked onto the ring by the boundary-snap pass (T-junction, ring
    gets split); (2) the closed orbit ring P[i:j+1], closure drift distributed
    linearly so end == start + k*pitch EXACTLY. The ring stays where the orbit
    is (no projection onto the far-away singularity) and is inserted as a
    BOUNDARY-type streamline, not a separatrix: it carries no singularity
    valence and the snap pass may split it. Clone prongs from the second
    cluster singularity converge to the SAME orbit, so near-duplicate rings are
    created only once (ring_dup_tol, set-Hausdorff mod pitch); tiled copies
    chain seamlessly across the strip (copy start node == neighbour end node)."""
    pitchn = getattr(mesh, "pitch_norm", None)
    if not (TILE_PERIODIC and pitchn):
        return
    n_sep = len(mesh.separatrices)
    n_b = len(mesh.streamlines) - n_sep

    def _resample(poly, n=80):
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        d = np.concatenate([[0.0], np.cumsum(seg)])
        if d[-1] < 1e-12:
            return None
        ss = np.linspace(0.0, 1.0, n)
        return np.column_stack([np.interp(ss, d / d[-1], poly[:, 0]),
                                np.interp(ss, d / d[-1], poly[:, 1])])

    def _ring_close(a, b):
        ra, rb = _resample(a), _resample(b)
        if ra is None or rb is None:
            return False
        best = np.inf
        k_c = int(round((ra[:, 0].mean() - rb[:, 0].mean()) / pitchn))
        for k in (k_c - 1, k_c, k_c + 1):
            d = np.linalg.norm(ra[:, None, :]
                               - (rb[None, :, :] + np.array([k * pitchn, 0.0])),
                               axis=2)
            best = min(best, max(d.min(axis=1).max(), d.min(axis=0).max()))
        return best < ring_dup_tol

    rings, docks, closed = [], [], 0
    for si in range(n_sep):
        P = np.asarray(mesh.streamlines[n_b + si], float)
        if P.ndim != 2 or len(P) < 50:
            continue
        T = np.gradient(P, axis=0)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-30)
        A = T @ T.T
        # "a couple of laps first": only allow the merge once the curve has
        # drifted at least min_wraps pitches in s from its start
        lapped = np.abs(P[:, 0] - P[0, 0]) >= min_wraps * pitchn
        hit = None
        for k in (1, -1, 2, -2):
            shift = np.array([k * pitchn, 0.0])
            D = np.linalg.norm(P[:, None, :] - (P[None, :, :] + shift), axis=2)
            cand = (D < tol) & (A > align)
            cand &= lapped[:, None]                      # j past min_wraps laps
            cand &= np.tri(len(P), k=-10, dtype=bool)   # j strictly after i
            js, iis = np.nonzero(cand)
            if len(js):
                j = int(js.min())                        # earliest converged hit
                i = int(iis[js == j][int(np.argmin(D[j, iis[js == j]]))])
                hit = (i, j, shift)
                break
        if hit is None:
            continue
        i, j, shift = hit
        R = P[i:j + 1].copy()
        # distribute the closure drift so R[-1] == R[0] + shift exactly
        err = (R[0] + shift) - R[-1]
        R = R + np.linspace(0.0, 1.0, len(R))[:, None] * err
        # place the ring in the central cover (start s wrapped into [0, pitch))
        R[:, 0] -= np.floor(R[0, 0] / pitchn) * pitchn
        # truncate the source curve to the finite prong docking on the ring:
        # end the prong EXACTLY on the ring (projection) and remember the dock
        # so partition() can emit the orthogonal crossing there (T -> X node)
        pe = P[i]
        best = (None, None, np.inf, None)
        for km in range(-MAX_WRAPS, MAX_WRAPS + 1):
            sh = np.array([km * pitchn, 0.0])
            seg, t, dist, proj = _project_to_polyline(pe - sh, R)
            if dist < best[2]:
                best = (km, (seg, t), dist, proj)
        km, _, _, proj = best
        dock = proj + np.array([km * pitchn, 0.0])       # in the prong's cover
        prong = np.vstack([P[:i], dock])
        v = P[i] - P[i - 1]
        v = v / (np.linalg.norm(v) + 1e-30)
        docks.append((dock, v))
        mesh.streamlines[n_b + si] = prong
        closed += 1
        if any(_ring_close(R, rr) for rr in rings):
            continue                                     # orbit already ringed
        rings.append(R)
    mesh.ring_docks = docks
    if rings:
        # insert as boundary-type streamlines (before the separatrix section)
        mesh.streamlines = (list(mesh.streamlines[:n_b]) + rings
                            + list(mesh.streamlines[n_b:]))
    if closed:
        print(f"[helix] closed {closed} helical prongs onto "
              f"{len(rings)} periodic orbit ring(s)")


def _emit_dock_crossings(sl, eps=0.012):
    """Turn each helix dock into an X-node (valence 4).

    A prong docking on an orbit ring is a T-junction (valence 3): the far side
    of the ring sees the dock node as a flat 180-degree point -> five-sided
    region -> dropped block (hole). The prong must NOT be continued straight
    (same cross family: it would spiral back onto the two-sided attractive
    limit orbit). Instead we emit the ORTHOGONAL cross family from the dock
    point on both sides; it runs in t towards inlet/outlet (no periodic trap).
    Dock valence then: ring edge 2 + prong 1 + crossing 1 = 4. Emitted BEFORE
    the snap pass so the crossing ends get docked/snapped normally and the ring
    is split at the dock. Near-duplicate crossings from clone-prong docks are
    left to the Hausdorff dedup."""
    mesh = sl.mesh
    docks = getattr(mesh, "ring_docks", None)
    if not docks:
        return
    added = 0
    for d, v in docks:
        d = np.asarray(d, float)
        v = np.asarray(v, float)
        for rot in (np.array([-v[1], v[0]]), np.array([v[1], -v[0]])):
            p0 = torch.tensor(d + eps * rot, dtype=torch.float)
            dir0 = torch.tensor(rot, dtype=torch.float)
            fi = sl.find_containing_face(p0, mesh)
            if fi is None:
                continue
            vec, _ = sl.get_best_cross_vector(p0, dir0, mesh, fi)
            if vec is None:
                continue
            curve = [d.copy(), (d + eps * rot).copy()]
            curve = sl.runge_kutta_heun_integrate_streamline(
                p0, vec, mesh, curve)
            if len(curve) < 3:
                continue
            mesh.streamlines.append(np.asarray(curve, float))
            mesh.separatrices.append({
                "coordinates": torch.tensor(d + eps * rot, dtype=torch.float),
                "vector": dir0,
                "singularity_coords": torch.tensor(d, dtype=torch.float),
                "face_id": -7000 - added,   # dock-crossing marker (no valence)
            })
            added += 1
    if added:
        print(f"[dock] emitted {added} orthogonal dock crossings (T -> X)")


def _drop_degenerate_corner_seps(mesh, min_len=0.1):
    """Drop outer-corner separatrices whose integrated streamline never left the
    seed neighbourhood.

    ``clean_separatrix._emanate_from_outer_corners`` seeds one interior arm per
    outer (corner_type==0) corner. At the two ACUTE parallelogram corners that arm
    runs almost along a wall and the RK integrator exits the mesh on the first
    step, yielding a 2-point stub (~0.04 long). Such a stub survives the snap pass
    (which only rejects <2-point curves) and reaches QuadFaceGenerator, whose block
    graph uses streamline ENDPOINTS as nodes: the stub becomes a graph edge to a
    dead-end (valence-1) node, so traverse_face slits the surrounding face region
    and the outer block is destroyed. The two OBTUSE corners emit genuine channel-
    crossing subdividers (arclen ~0.3-0.5), which we keep. Filter is restricted to
    corner-origin separatrices (``corner_origin`` flag) so real but short blade-tip
    prongs are never touched; min_len=0.1 sits safely between stub and subdivider.
    """
    seps = mesh.separatrices
    n_sep = len(seps)
    n_boundary = len(mesh.streamlines) - n_sep
    keep_sl = list(mesh.streamlines[:n_boundary])
    keep_dicts = []
    dropped = 0
    for i in range(n_sep):
        sl = mesh.streamlines[n_boundary + i]
        if seps[i].get("corner_origin"):
            s = np.asarray(sl, float)
            arclen = (float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
                      if s.ndim == 2 and len(s) >= 2 else 0.0)
            if arclen < min_len:
                dropped += 1
                continue
        keep_sl.append(sl)
        keep_dicts.append(seps[i])
    mesh.streamlines = keep_sl
    mesh.separatrices = keep_dicts
    if dropped:
        print(f"[corner-drop] removed {dropped} degenerate corner-stub separatrices")


def _validate_separatrix_counts(mesh):
    """Check the per-singularity separatrix-count invariant (Kowalski 2015): a
    field singularity of representation index +1 must keep EXACTLY 3 incident
    separatrices, index -1 EXACTLY 5. A violation after postprocessing is the
    direct cause of a non-four-sided region (irregular interior block node /
    Euler defect). Returns (all_ok, report_lines)."""
    sing_fids = list(getattr(mesh, "singularities_coords", {}).keys())
    if not sing_fids:
        return True, ["separatrix-count: no field singularities"]
    n_sing = len(sing_fids)
    got = {k: 0 for k in range(n_sing)}
    for sd in mesh.separatrices:
        for key in ("origin_sing", "target_sing"):
            kk = sd.get(key) if isinstance(sd, dict) else None
            if kk is not None and 0 <= kk < n_sing:
                got[kk] += 1
    all_ok = True
    lines = ["separatrix-count per singularity (Kowalski 3/5 invariant):"]
    for k, fid in enumerate(sing_fids):
        idx = int(mesh.singularities[fid].item())
        exp = int(mesh.expected_separatrices.get(fid, 4))
        g = got[k]
        ok = (g == exp)
        all_ok = all_ok and ok
        lines.append(f"  fid={fid} idx={idx:+d} expected={exp} got={g} "
                     f"{'OK' if ok else 'MISMATCH'}")
    lines.append(f"separatrix-count invariant: {'PASS' if all_ok else 'FAIL'}")
    return all_ok, lines


def validate(block_mesh, mesh, out_path=None):
    """Run domain_partition's QuadPartitionValidator on the finished block_mesh.

    Validation is only defined AFTER postprocessing has produced quad faces; if
    postproc yielded nothing, that is itself a validation failure (reported, not
    crashed). frame_field is the per-node cross representation mesh.u (cos4t/sin4t),
    needed for the Kowalski singularity-efficiency metric."""
    lines = []
    if (block_mesh is None or block_mesh.faces is None
            or block_mesh.faces.numel() == 0):
        lines.append("VALIDATION FAIL: postprocessing produced no quad blocks.")
        report = "\n".join(lines)
        if out_path:
            Path(out_path).write_text(report)
        print(report)
        return None

    v = QuadPartitionValidator(block_mesh, mesh, frame_field=mesh.u, strict=True)
    is_valid = v.is_valid()
    qs = v.quality_score()
    soft = v.passes_soft_thresholds()
    diag = v.diagnostics()

    lines.append(f"is_valid (hard checks): {is_valid}")
    lines.append(f"passes_soft_thresholds: {soft}")
    lines.append(f"blocks: {block_mesh.faces.shape[1]}  "
                 f"corners: {block_mesh.x.shape[0]}")
    exp = getattr(v, "_expected_singularities", None)
    act = getattr(v, "_actual_singularities", None)
    if exp is not None:
        eff = qs.get("singularity_efficiency")
        lines.append(f"singularities expected/actual: {exp}/{act}  "
                     f"efficiency: {eff}")
    lines.append("quality_score:")
    for k, val in qs.items():
        lines.append(f"  {k}: {val}")
    lines.append("diagnostics:")
    lines.extend(f"  - {d}" for d in (diag or ["(none)"]))

    sep_ok, sep_lines = _validate_separatrix_counts(mesh)
    lines.extend(sep_lines)

    report = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(report)
        print(f"wrote {out_path}")
    print(report)
    return v


def _termination_nodes(mesh):
    """Targets a separatrix may end on: singularities + c0 corners."""
    pts = [np.array(c) for c in mesh.singularities_coords.values()]
    corners = mesh.x[mesh.x[:, 2] == 0, 0:2].numpy()
    pts.extend(list(corners))
    return np.array(pts) if pts else np.zeros((0, 2))


def _project_to_polyline(p, poly):
    """Closest point on a polyline. Returns (seg_index, t, dist, proj_point)."""
    best = (None, 0.0, np.inf, None)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = ab @ ab
        t = 0.0 if L2 < 1e-18 else np.clip((p - a) @ ab / L2, 0.0, 1.0)
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best[2]:
            best = (i, t, d, proj)
    return best


def _min_boundary_dist(p, boundary):
    """Distance from point p to the nearest boundary polyline (incl. blade loop)."""
    p = np.asarray(p, float)
    best = np.inf
    for poly in boundary:
        d = _project_to_polyline(p, poly)[2]
        if d < best:
            best = d
    return best


def _snap_separatrix_endpoints(mesh, radius=0.045, bnd_radius=0.05, end_frac=0.3):
    """Kowalski post-processing: every separatrix must end on a singularity, a
    c0 corner, or the boundary dOmega (outer *and* the inner blade loop). The RK
    integrator drifts past targets and the block graph only treats streamline
    *endpoints* as nodes, so we:
      (1) truncate+snap a separatrix to the first point-target (singularity/
          corner) it enters, else
      (2) snap its dangling end onto the nearest boundary polyline and *split*
          that boundary streamline at the snap point, so the new T-junction is a
          shared graph node and blocks can close against it (notably the blade).
    """
    nodes = _termination_nodes(mesh)
    n_sing = len(mesh.singularities_coords)  # nodes[:n_sing] are singularities
    n_boundary = len(mesh.streamlines) - len(mesh.separatrices)
    boundary = [np.asarray(s, float) for s in mesh.streamlines[:n_boundary]]
    seps = [np.asarray(s, float) for s in mesh.streamlines[n_boundary:]]
    sep_dicts = list(mesh.separatrices)

    # Periodic wrap (Phase 2b): separatrix ends may now lie in the neighbour
    # passage (cover coords). Tile the point targets and boundary polylines by
    # +/- one pitch so those ends snap/split correctly; per-copy shifts let us
    # map every hit back to its central entity.
    n_nodes0, n_bnd0 = len(nodes), len(boundary)
    pitchn = getattr(mesh, "pitch_norm", None)
    tiled = bool(TILE_PERIODIC and pitchn and n_nodes0)
    node_shift = np.zeros(max(n_nodes0, 1))
    bnd_shift = [0.0] * n_bnd0
    if tiled:
        # cover as many wraps as the integrated curves actually reach
        s_ext = max(abs(float(np.min([s[:, 0].min() for s in seps if s.ndim == 2]
                                     or [0.0]))),
                    abs(float(np.max([s[:, 0].max() for s in seps if s.ndim == 2]
                                     or [1.0])) - 1.0)) if seps else 0.0
        K = min(MAX_WRAPS, int(np.ceil(s_ext / pitchn)) + 1)
        for k in [kk for k0 in range(1, K + 1) for kk in (k0, -k0)]:
            shv = np.array([k * pitchn, 0.0])
            nodes = np.vstack([nodes, nodes[:n_nodes0] + shv])
            node_shift = np.concatenate([node_shift,
                                         np.full(n_nodes0, k * pitchn)])
            boundary = boundary + [b + shv for b in boundary[:n_bnd0]]
            bnd_shift = bnd_shift + [k * pitchn] * n_bnd0

    def _origin_sing(si):
        """node index of a separatrix's origin if it is a field singularity."""
        if si >= len(sep_dicts) or int(sep_dicts[si].get("face_id", -1)) < 0:
            return None  # blade-tip / corner origin, not a field singularity
        oc = np.asarray(sep_dicts[si]["singularity_coords"], float)
        if n_sing == 0:
            return None
        k = int(np.argmin(np.linalg.norm(nodes[:n_sing] - oc, axis=1)))
        return k

    # --- pass 1: snap each separatrix, record metadata (no boundary split yet) -
    recs = []  # per-sep: dict(poly, origin_sing, target_sing, bnd=(bi,segpos,proj))
    n_pt, n_bnd = 0, 0
    for si, s in enumerate(seps):
        rec = {"poly": s, "origin_sing": _origin_sing(si),
               "target_sing": None, "bnd": None, "tshift": 0.0}
        if s.ndim != 2 or len(s) < 2:
            recs.append(rec)
            continue
        origin = s[0]
        origin_node = int(np.argmin(np.linalg.norm(nodes - origin, axis=1))) \
            if len(nodes) else -1
        # (1) point-target snapping along the path.
        # Blade-vs-neighbour guard: the two tip singularities of a cluster sit
        # ~0.03 apart (below `radius`). A short separatrix emanating from one tip
        # toward the blade ends ON the blade loop right between the two tips, but
        # the neighbouring tip singularity is within `radius` of that end, so the
        # naive nearest-node snap truncates it at the neighbour instead of letting
        # it dock on the blade. Skip such a singularity snap when the approach
        # point is closer to a boundary than to that singularity, lies within
        # bnd_radius of the boundary, AND is near the END of the separatrix
        # (little arclength remaining). The last clause is essential: a runaway
        # multi-wrap spiral around the blade also hugs the blade, but it grazes
        # the neighbour at its START (lots of path remaining) -- there we DO keep
        # the sing snap so the spiral is discarded, not kept.
        seg_all = np.linalg.norm(np.diff(s, axis=0), axis=1)
        total_len = float(seg_all.sum())
        cut = False
        for j in range(1, len(s)):
            if len(nodes) == 0:
                break
            d = np.linalg.norm(nodes - s[j], axis=1)
            k = int(np.argmin(d))
            k_c = k % n_nodes0 if n_nodes0 else k   # central node of a copy
            if d[k] < radius and k != origin_node:
                if k_c < n_sing and total_len > 1e-9:
                    dbl = _min_boundary_dist(s[j], boundary)
                    rem = float(seg_all[j:].sum()) / total_len
                    if dbl < d[k] and dbl < bnd_radius and rem < end_frac:
                        continue               # terminates on the blade, not S_k
                s = np.vstack([s[:j], nodes[k]])
                cut = True
                n_pt += 1
                if k_c < n_sing:               # snapped onto a field singularity
                    rec["target_sing"] = k_c   # central id; cover offset below
                    rec["tshift"] = float(node_shift[k])
                break
        # (2) boundary snapping of the (still dangling) end
        if not cut:
            end = s[-1]
            best = (None, None, np.inf, None)
            for bi, poly in enumerate(boundary):
                seg, t, dist, proj = _project_to_polyline(end, poly)
                if dist < best[2]:
                    best = (bi, (seg, t), dist, proj)
            if best[0] is not None and best[2] < bnd_radius:
                bi, (seg, t), _, proj = best
                s = np.vstack([s[:-1], proj])
                # map a hit on a +/-pitch boundary COPY back to the central
                # polyline: split the central one at proj - shift; its tiled
                # copy then passes exactly through proj, so the T-junction node
                # matches the (cover-coordinate) separatrix end downstream.
                bi_c = bi % n_bnd0 if n_bnd0 else bi
                proj_c = proj - np.array([bnd_shift[bi], 0.0])
                rec["bnd"] = (bi_c, seg + t, proj_c)
                n_bnd += 1
        rec["poly"] = s
        recs.append(rec)

    # --- blend separatrices joining two singularities (Kowalski 2015, sec 4.3) -
    # A separatrix S0->S1 between two field singularities is integrated TWICE:
    # once from S0 (rec i: origin S0, target S1) and once from S1 (rec j: origin
    # S1, target S0). Round-off makes the two paths differ. Kowalski Eq 28 does
    # NOT delete one of them; it linearly BLENDS them into a single, more accurate
    # curve, trusting each path near the singularity it spawned from:
    #     gamma_b(s) = (1 - s) * gamma1(s) + s * gamma2(s),   s in [0, 1]
    # with gamma1: S0->S1 and gamma2: S1->S0 (read reversed, i.e. S0->S1). We keep
    # the blended curve as rec i and drop rec j. This yields exactly one clean
    # separatrix per singularity-singularity connection (correct 3/5 valence) with
    # no duplicate and no residual integration error.
    def _resample(poly, n):
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            return None
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        d = np.concatenate([[0.0], np.cumsum(seg)])
        if d[-1] < 1e-12:
            return None
        ss = np.linspace(0.0, 1.0, n)
        return np.column_stack([np.interp(ss, d / d[-1], poly[:, 0]),
                                np.interp(ss, d / d[-1], poly[:, 1])])

    N_BLEND = 60
    drop = set()
    for i, ri in enumerate(recs):
        if i in drop:
            continue
        A, B = ri["origin_sing"], ri["target_sing"]
        if A is None or B is None or A == B:
            continue
        # matching reverse separatrix: origin B, target A (Case 1, Fig 8a).
        # With periodic wrap the same physical connection may cross the seam:
        # emitted as A -> B+pitch (tshift +p) one way and B -> A-pitch (-p) the
        # other; the pair is consistent iff the cover offsets cancel.
        j = next((k for k, rk in enumerate(recs)
                  if k not in drop and k != i
                  and rk["origin_sing"] == B and rk["target_sing"] == A
                  and abs(rk["tshift"] + ri["tshift"]) < 1e-6), None)
        if j is None:
            continue
        g1 = _resample(ri["poly"], N_BLEND)          # S0 -> S1
        g2 = _resample(recs[j]["poly"], N_BLEND)     # S1 -> S0
        if g1 is None or g2 is None:
            continue
        # read reversed as S0 -> S1, shifted into g1's cover
        g2 = g2[::-1] + np.array([ri["tshift"], 0.0])
        s = np.linspace(0.0, 1.0, N_BLEND)[:, None]
        ri["poly"] = (1.0 - s) * g1 + s * g2         # Kowalski Eq 28 / Xiao Eq 10
        drop.add(j)

    # --- Xiao 2020 Case 2 (Fig 8b): geometric reverse-duplicate wraps -----------
    # A wrap along one side of the blade connects a TE tip to an LE tip. Because
    # each tip is a *cluster* of two singularities ~0.03 apart, the same physical
    # wrap gets emitted from BOTH ends but the two integrations snap to DIFFERENT
    # cluster-neighbours (e.g. S0->S2 one way, S3->S0 the other). The Case-1 blend
    # above only matches the mutual A->B / B->A pair, so it misses this: the two
    # copies survive and over-count their shared tip (Kowalski 3/5 invariant then
    # fails, e.g. a -1 tip gets 6 separatrices instead of 5). Detect the duplicate
    # geometrically (near-identical curve, orientation-independent Hausdorff) and
    # drop ONE copy -- specifically the copy whose removal keeps BOTH its endpoint
    # singularities at or above their required count (Xiao/Kowalski count-safe
    # dedup). Blend the survivor with the reversed dropped copy for accuracy.
    sfids = list(mesh.singularities_coords.keys())

    def _expected(k):
        if k is None or k >= len(sfids):
            return 4
        fid = sfids[k]
        idx = int(mesh.singularities[fid].item())
        return int(getattr(mesh, "expected_separatrices", {}).get(
            fid, 3 if idx == 1 else 5 if idx == -1 else 4))

    def _got(exclude=()):
        g = {}
        for ii, rr in enumerate(recs):
            if ii in drop or ii in exclude:
                continue
            for key in ("origin_sing", "target_sing"):
                kk = rr[key]
                if kk is not None:
                    g[kk] = g.get(kk, 0) + 1
        return g

    def _set_hausdorff(a, b):
        # orientation-independent max of nearest-neighbour distances
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return max(d.min(axis=1).max(), d.min(axis=0).max())

    def _set_hausdorff_p(a, b):
        # periodic variant: duplicates of the same physical curve may live in
        # different covers (offset by whole pitches) -- compare modulo the pitch.
        best = _set_hausdorff(a, b)
        if tiled:
            k_c = int(round((a[:, 0].mean() - b[:, 0].mean()) / pitchn))
            for k in {k_c - 1, k_c, k_c + 1} - {0}:
                best = min(best,
                           _set_hausdorff(a, b + np.array([k * pitchn, 0.0])))
        return best

    sing_node_ids = [k for k in range(len(nodes))
                     if (k % n_nodes0 if n_nodes0 else k) < n_sing]

    def _snap_sing(p):
        # central id of the field singularity within `radius` of p, else None
        best, bid = radius, None
        for k in sing_node_ids:
            d = float(np.linalg.norm(nodes[k] - p))
            if d < best:
                best, bid = d, k % n_nodes0 if n_nodes0 else k
        return bid

    def _ends_match(a, b, loose=0.06):
        # Duplicates of one physical curve must share their endpoints (either
        # orientation, modulo the pitch). Exact case: both ends within the
        # snap radius. Cluster-wrap case (comment above): one end tight, the
        # other pair up to a tip-cluster spread apart -- legitimate only when
        # those two ends sit on two DIFFERENT singularities that also differ
        # from the tight end, i.e. the same wrap snapped to different
        # cluster-neighbours. Two arms of different singularities converging
        # onto a shared tail (or a stub next to a through-arm) fail this:
        # after point-snap truncation the endpoints encode arm ownership.
        shifts = [0.0]
        if tiled:
            k_c = int(round((a[:, 0].mean() - b[:, 0].mean()) / pitchn))
            shifts = [k * pitchn for k in {k_c - 1, k_c, k_c + 1}]
        for sh in shifts:
            b0, b1 = b[0] + np.array([sh, 0.0]), b[-1] + np.array([sh, 0.0])
            for p, q in ((b0, b1), (b1, b0)):
                d0 = np.linalg.norm(a[0] - p)
                d1 = np.linalg.norm(a[-1] - q)
                if d0 < radius and d1 < radius:
                    return True
                if d0 < radius and d1 < loose:
                    tight, la, lb = a[0], a[-1], q
                elif d1 < radius and d0 < loose:
                    tight, la, lb = a[-1], a[0], p
                else:
                    continue
                sa, sb, st = _snap_sing(la), _snap_sing(lb), _snap_sing(tight)
                if sa is not None and sb is not None \
                        and sa != sb and sa != st and sb != st:
                    return True
        return False

    DUP_TOL = 0.08
    # candidates: ALL separatrices. With periodic wrap the same physical curve
    # can be integrated several times in different covers: the (1,1) corner arm
    # is the (0.407,1) arm shifted one pitch (identical physical point), and a
    # blade wrap is emitted from both cluster ends. Near-coincident survivors
    # tile into stacks of duplicates and produce sliver triangles in the block
    # graph. Count-safety below still protects field-singularity valences;
    # corner/boundary endpoints are unconstrained.
    sing_recs = [i for i, r in enumerate(recs)
                 if i not in drop
                 and not (r["origin_sing"] is not None
                          and r["origin_sing"] == r["target_sing"])]
    for a_i in range(len(sing_recs)):
        i = sing_recs[a_i]
        if i in drop:
            continue
        gi = _resample(recs[i]["poly"], N_BLEND)
        if gi is None:
            continue
        for b_i in range(a_i + 1, len(sing_recs)):
            j = sing_recs[b_i]
            if j in drop:
                continue
            gj = _resample(recs[j]["poly"], N_BLEND)
            if gj is None or _set_hausdorff_p(gi, gj) > DUP_TOL:
                continue
            if not _ends_match(gi, gj):
                continue
            got = _got()
            exp = {k: _expected(k) for k in got}

            def _safe_to_drop(m):
                for key in ("origin_sing", "target_sing"):
                    k = recs[m][key]
                    if k is not None and got.get(k, 0) - 1 < exp.get(k, 4):
                        return False
                return True

            victim = i if _safe_to_drop(i) else (j if _safe_to_drop(j) else None)
            if victim is None:
                continue
            keep = j if victim == i else i
            gk = _resample(recs[keep]["poly"], N_BLEND)
            gv = _resample(recs[victim]["poly"], N_BLEND)
            # blend survivor with dropped copy only when both live in the SAME
            # cover (unshifted-close); a pitch-offset duplicate is just dropped.
            if gk is not None and gv is not None \
                    and _set_hausdorff(gk, gv) <= DUP_TOL:
                if _set_hausdorff(gk[:1], gv[:1]) > _set_hausdorff(gk[:1], gv[-1:]):
                    gv = gv[::-1]              # align victim to survivor direction
                ss = np.linspace(0.0, 1.0, N_BLEND)[:, None]
                recs[keep]["poly"] = (1.0 - ss) * gk + ss * gv
            drop.add(victim)
            if victim == i:
                break                          # i gone, move to next survivor

    kept = [i for i in range(len(recs)) if i not in drop]

    # stamp origin/target singularity back onto the surviving separatrix dicts
    # so the count validator (and any downstream graph) can read the connectivity
    for i in kept:
        sep_dicts[i]["origin_sing"] = recs[i]["origin_sing"]
        sep_dicts[i]["target_sing"] = recs[i]["target_sing"]

    # --- pass 2: boundary T-junction splits, only from KEPT separatrices -------
    splits = {i: [] for i in range(n_boundary)}
    for i in kept:
        if recs[i]["bnd"] is not None:
            bi, segpos, proj = recs[i]["bnd"]
            splits[bi].append((segpos, proj))

    new_boundary = []
    for bi, poly in enumerate(boundary[:n_bnd0]):   # central originals only
        pts = sorted(splits[bi])
        if not pts:
            new_boundary.append(poly)
            continue
        cuts = []
        cur = [poly[0]]
        ptr = 0
        for i in range(len(poly) - 1):
            cur.append(poly[i + 1])
            while ptr < len(pts) and seg_floor(pts[ptr][0]) == i:
                proj = pts[ptr][1]
                cur[-1] = proj  # end this sub-streamline at the T-junction
                cuts.append(np.array(cur))
                cur = [proj, poly[i + 1]]
                ptr += 1
        cuts.append(np.array(cur))
        new_boundary.extend(c for c in cuts if len(c) >= 2)

    mesh.streamlines = new_boundary + [recs[i]["poly"] for i in kept]
    mesh.separatrices = [sep_dicts[i] for i in kept]
    print(f"[snap] point-snapped {n_pt}, boundary-snapped {n_bnd}; "
          f"deduped {len(drop)} matching prongs; "
          f"boundary {n_boundary}->{len(new_boundary)} segments, "
          f"separatrices {len(seps)}->{len(kept)}")


def seg_floor(x):
    return int(np.floor(x))


def _plot_separatrices(mesh, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()
    n_b = len(mesh.streamlines) - len(mesh.separatrices)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="0.92")
    for i, s in enumerate(mesh.streamlines):
        s = np.asarray(s, dtype=float)
        if s.ndim != 2 or len(s) < 2:
            continue
        ax.plot(s[:, 0], s[:, 1], "0.4" if i < n_b else "C3",
                lw=1.0 if i < n_b else 1.4)
    sc = xy[tris[mesh.singularities.numpy() != 0]].mean(axis=1)
    ax.scatter(sc[:, 0], sc[:, 1], c="blue", s=70, zorder=6, label="singularity")
    cm = mesh.x[:, 2].numpy() == 0
    ax.scatter(xy[cm, 0], xy[cm, 1], c="green", s=60, marker="^", zorder=6,
               label="c0 corner")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{len(mesh.separatrices)} separatrices, "
                 f"{int((mesh.singularities.numpy()!=0).sum())} singularities")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


def _block_annotations(bx, bf, ratio_max=10.0):
    """Per-block-mesh validation annotations for plotting:
    inverted quads, interior nodes with valence != 4, high-aspect quads."""
    # edge topology
    from collections import defaultdict
    cnt = defaultdict(int)
    deg = np.zeros(len(bx), dtype=int)
    for quad in bf:
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]
            cnt[(min(a, b), max(a, b))] += 1
    for (a, b), c in cnt.items():
        deg[a] += 1
        deg[b] += 1
    boundary_nodes = set()
    for (a, b), c in cnt.items():
        if c == 1:
            boundary_nodes.add(a)
            boundary_nodes.add(b)
    irregular = [i for i in range(len(bx))
                 if i not in boundary_nodes and deg[i] != 4]
    inverted, high_aspect = [], []
    for qi, quad in enumerate(bf):
        p = bx[quad]
        # signed area (shoelace); negative => inverted ordering
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                            - np.roll(p[:, 0], -1) * p[:, 1])
        if area <= 0:
            inverted.append(qi)
        elens = [np.linalg.norm(p[(k + 1) % 4] - p[k]) for k in range(4)]
        if min(elens) > 1e-12 and max(elens) / min(elens) > ratio_max:
            high_aspect.append(qi)
    return irregular, inverted, high_aspect


def _plot_blocks(block_mesh, mesh, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = mesh.x[:, 0:2].numpy()
    tris = mesh.faces.T.numpy()
    bx = block_mesh.x.numpy()
    bf = block_mesh.faces.numpy().T  # (B,4)
    irregular, inverted, high_aspect = _block_annotations(bx, bf)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.triplot(xy[:, 0], xy[:, 1], tris, lw=0.1, color="0.92")
    for qi, quad in enumerate(bf):
        ring = bx[list(quad) + [quad[0]]]
        if qi in inverted:
            ax.fill(ring[:, 0], ring[:, 1], color="red", alpha=0.4)
            ax.plot(ring[:, 0], ring[:, 1], "red", lw=2.0)
        elif qi in high_aspect:
            ax.fill(ring[:, 0], ring[:, 1], color="orange", alpha=0.3)
            ax.plot(ring[:, 0], ring[:, 1], "darkorange", lw=1.8)
        else:
            ax.fill(ring[:, 0], ring[:, 1], alpha=0.25)
            ax.plot(ring[:, 0], ring[:, 1], "C0", lw=1.3)
    ax.scatter(bx[:, 0], bx[:, 1], c="k", s=8, zorder=5)
    if irregular:
        ax.scatter(bx[irregular, 0], bx[irregular, 1], facecolors="none",
                   edgecolors="red", s=160, linewidths=2.0, zorder=6,
                   label=f"irregular interior node (valence!=4): {len(irregular)}")
    if high_aspect:
        ax.plot([], [], "darkorange", lw=1.8,
                label=f"aspect>10: {len(high_aspect)}")
    if inverted:
        ax.plot([], [], "red", lw=2.0, label=f"inverted: {len(inverted)}")
    if irregular or high_aspect or inverted:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"Hub quad block partition: {bf.shape[0]} blocks  "
                 f"(irregular={len(irregular)}, inverted={len(inverted)}, "
                 f"aspect>10={len(high_aspect)})")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


