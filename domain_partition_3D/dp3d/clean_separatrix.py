#!/usr/bin/env python3
"""
A robust replacement for domain_partition's SeparatrixGenerator.

domain_partition's v1/v2 separatrix generators are buggy and were never
exercised on meshes with real singularities (the NACA data generation filters
to near-zero-singularity fields). We only need them to populate
``mesh.separatrices`` in the format that ``StreamlineGenerator_v2.get_streamlines``
consumes; that RK/Heun integrator itself is sound and is reused unchanged.

Emanation directions are found by walking a small circle around each singularity
and locating the angles where the (4-fold) cross field aligns radially - those
are exactly the separatrix departure directions. Boundary corner (dim-0) nodes
emit separatrices along the interior cross direction.

Interface (matches what StreamlineGenerator_v2 expects):
    mesh.singularities          (F,)   Poincare index per face   [reused logic]
    mesh.singularities_coords   {face: [x,y]}
    mesh.expected_separatrices  {face: count}
    mesh.separatrices           [ {coordinates, vector, singularity_coords} ]
"""

import numpy as np
import torch

# Toggle Kowalski singularity pair annihilation (set via partition_surface).
# Pure deletion of a +k/-k pair over-simplifies and can break the blade wrap;
# kept switchable for comparison. Default off until the bridging-streamline
# variant is in place.
ANNIHILATE_PAIRS = False

# Experiment (Step 0): emit separatrices from the OUTER (corner_type==0) domain
# corners along their interior cross direction, to test whether that alone tiles
# the outer channels into quads (vs the heavier periodic-tiling approach).
EMANATE_OUTER_CORNERS = False


def set_annihilate_pairs(flag):
    global ANNIHILATE_PAIRS
    ANNIHILATE_PAIRS = bool(flag)


def set_emanate_outer_corners(flag):
    global EMANATE_OUTER_CORNERS
    EMANATE_OUTER_CORNERS = bool(flag)


def _poincare_indices(mesh):
    u = mesh.u
    ang = torch.atan2(u[:, 1], u[:, 0])
    f = mesh.faces
    a = ang[f[0]]
    b = ang[f[1]]
    c = ang[f[2]]
    def wrap(d):
        return torch.remainder(d + torch.pi, 2 * torch.pi) - torch.pi
    s = wrap(b - a) + wrap(c - b) + wrap(a - c)
    return torch.round(s / (2 * torch.pi))


def _singularity_coords(mesh, face_id):
    face = mesh.faces[:, face_id]
    vecs = mesh.u[face]
    nodes = mesh.x[face, 0:2]
    A = torch.stack([vecs[0] - vecs[2], vecs[1] - vecs[2]], dim=1)
    try:
        co = torch.linalg.solve(A, -vecs[2])
        if (co >= -0.2).all() and co.sum() <= 1.2:
            return nodes[2] + co[0] * (nodes[0] - nodes[2]) + co[1] * (nodes[1] - nodes[2])
    except RuntimeError:
        pass
    return nodes.mean(dim=0)


class CleanSeparatrixGenerator:
    def __init__(self, mesh, corner_merge_tol=0.06):
        self.mesh = mesh
        self.corner_merge_tol = corner_merge_tol
        self._build_locator()
        # If prescribed singularities were already injected (e.g. from Hub
        # template), keep them instead of recomputing from the field.
        if getattr(mesh, 'singularities_coords', None):
            self._ensure_expected()
        else:
            mesh.singularities = _poincare_indices(mesh)
            self._coords_and_expected()
        self._delete_corner_singularities()
        if ANNIHILATE_PAIRS:
            self._annihilate_pairs()
        seps = self._emanate_from_singularities()
        seps += self._emanate_from_blade_tips()
        if EMANATE_OUTER_CORNERS:
            seps += self._emanate_from_outer_corners()
        mesh.separatrices = seps
        self.found_separatrices = seps

    def _delete_corner_singularities(self):
        """Drop singularities that sit right next to an OUTER (corner_type==0)
        boundary corner.

        Slanted (non-90deg) outer corners get a bisector Dirichlet cross that
        conflicts with the wall-aligned representative of the adjacent boundary
        nodes, forcing a spurious singularity next to each corner; those are
        artifacts and get removed.

        Blade-tip (corner_type==1, LE/TE) corners are NOT swept here: with a clean
        (e.g. periodic) field the genuine topological singularity sits exactly at
        the tip and must be KEPT to drive the O-grid-around-blade partition
        (Kowalski 2015). We therefore restrict the deletion to outer corners."""
        mesh = self.mesh
        ct = getattr(mesh, "corner_type", None)
        if ct is not None:
            ct = ct.numpy() if hasattr(ct, "numpy") else np.asarray(ct)
            corner_xy = self.nodes[ct == 0]      # outer corners only
        else:
            corner_xy = self.nodes[mesh.x[:, 2].numpy() == 0]
        if len(corner_xy) == 0:
            return
        removed = 0
        for fid in list(mesh.singularities_coords.keys()):
            c = np.array(mesh.singularities_coords[fid])
            if np.min(np.linalg.norm(corner_xy - c, axis=1)) < self.corner_merge_tol:
                mesh.singularities[fid] = 0
                del mesh.singularities_coords[fid]
                mesh.expected_separatrices.pop(fid, None)
                removed += 1
        print(f"[clean_sep] deleted {removed} near-corner singularities, "
              f"{len(mesh.singularities_coords)} remain")

    def _annihilate_pairs(self, max_dist=0.15):
        """Kowalski-style singularity pair annihilation.

        A +k and -k cross singularity close together cancel: the field between
        them is regular up to a removable branch. We remove mutually-nearest
        opposite-index pairs within max_dist so neither emits separatrices,
        collapsing the spurious irregular block node they would force. (Net
        topological index is fixed by the boundary holonomy, so only pairs that
        actually cancel are removed; an index imbalance is left untouched.)"""
        mesh = self.mesh
        items = [(fid, int(mesh.singularities[fid]), np.array(c, float))
                 for fid, c in mesh.singularities_coords.items()]
        n = len(items)
        if n < 2:
            return

        def nearest_opposite(i):
            fi, ii, ci = items[i]
            best, bestd = None, np.inf
            for j in range(n):
                fj, ij, cj = items[j]
                if j == i or ij != -ii:
                    continue
                d = float(np.linalg.norm(ci - cj))
                if d < bestd:
                    bestd, best = d, j
            return best, bestd

        nn = [nearest_opposite(i) for i in range(n)]
        removed = set()
        for i in range(n):
            j, d = nn[i]
            if j is None or d > max_dist:
                continue
            if nn[j][0] == i:  # mutual nearest opposite-index pair
                removed.add(items[i][0])
                removed.add(items[j][0])
        for fid in removed:
            mesh.singularities[fid] = 0
            mesh.singularities_coords.pop(fid, None)
            mesh.expected_separatrices.pop(fid, None)
        print(f"[clean_sep] annihilated {len(removed)} singularities in "
              f"opposite-index pairs, {len(mesh.singularities_coords)} remain")

    # --- point location ---------------------------------------------------
    def _build_locator(self):
        self.nodes = self.mesh.x[:, 0:2].numpy().astype(np.float64)
        self.faces = self.mesh.faces.numpy().T  # (F,3)
        self.n2f = self.mesh.nodes_faces_ids
        self.u = self.mesh.u.numpy().astype(np.float64)

    def _find_face(self, p):
        d = np.sum((self.nodes - p) ** 2, axis=1)
        for fi in self.n2f.get(int(np.argmin(d)), []):
            tri = self.faces[fi]
            if self._bary_inside(p, self.nodes[tri]):
                return fi
        return None

    @staticmethod
    def _bary(p, tri):
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = p - tri[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-18:
            return None
        a = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        b = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return np.array([1 - a - b, a, b])

    def _bary_inside(self, p, tri):
        bc = self._bary(p, tri)
        return bc is not None and (bc >= -1e-9).all()

    def _cross_dirs(self, p):
        """Return the 4 unit cross directions of the field at point p (or None)."""
        fi = self._find_face(p)
        if fi is None:
            return None
        tri = self.faces[fi]
        bc = self._bary(p, self.nodes[tri])
        if bc is None:
            return None
        iv = bc @ self.u[tri]
        n = np.linalg.norm(iv)
        if n < 1e-9:
            return None
        base = np.arctan2(iv[1], iv[0]) / 4.0
        return base + np.arange(4) * (np.pi / 2)

    # --- singularity metadata --------------------------------------------
    def _coords_and_expected(self):
        mesh = self.mesh
        mesh.singularities_coords = {}
        mesh.expected_separatrices = {}
        face_ids = torch.where(mesh.singularities != 0)[0]
        for fid in face_ids.tolist():
            mesh.singularities_coords[fid] = _singularity_coords(mesh, fid).tolist()
            idx = int(mesh.singularities[fid].item())
            mesh.expected_separatrices[fid] = 3 if idx == 1 else (5 if idx == -1 else abs(4 * idx - 1))

    def _ensure_expected(self):
        """Ensure expected_separatrices is set for existing singularities."""
        mesh = self.mesh
        if not hasattr(mesh, 'expected_separatrices') or not mesh.expected_separatrices:
            mesh.expected_separatrices = {}
            for fid in mesh.singularities_coords:
                idx = int(mesh.singularities[fid].item())
                mesh.expected_separatrices[fid] = 3 if idx == 1 else (5 if idx == -1 else abs(4 * idx - 1))

    def _local_scale(self, fid):
        tri = self.nodes[self.faces[fid]]
        return float(np.mean([np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3)]))

    # --- emanation --------------------------------------------------------
    def _emanate_from_singularities(self, n_samples=720):
        seps = []
        mesh = self.mesh
        alphas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        for fid, coords in mesh.singularities_coords.items():
            c = np.array(coords)
            eps = 1.2 * self._local_scale(fid)
            g = np.full(n_samples, np.pi)
            for i, al in enumerate(alphas):
                p = c + eps * np.array([np.cos(al), np.sin(al)])
                cd = self._cross_dirs(p)
                if cd is None:
                    continue
                # angular distance from radial direction al to nearest cross dir
                diff = np.angle(np.exp(1j * (cd - al)))
                g[i] = np.min(np.abs(diff))
            # A +1 (geometric +1/4) singularity has EXACTLY 3 separatrices, a -1
            # (-1/4) one EXACTLY 5 (Kowalski 2015, sec 3.2.3 / Fig 8). The
            # partition is only four-sided (Prop 9) if every singularity emits its
            # full prong set, so we enforce the count here instead of taking
            # however many minima a fixed threshold yields.
            expected = mesh.expected_separatrices.get(fid)
            dirs = self._emanation_dirs(alphas, g, expected)
            for al in dirs:
                d = np.array([np.cos(al), np.sin(al)])
                seps.append({
                    "coordinates": torch.tensor(c + eps * d, dtype=torch.float),
                    "vector": torch.tensor(d, dtype=torch.float),
                    "singularity_coords": torch.tensor(c, dtype=torch.float),
                    "face_id": fid,
                })
        return seps

    def _emanation_dirs(self, alphas, g, expected):
        """Return exactly ``expected`` separatrix departure angles for a
        singularity, given the radial-misalignment profile ``g(alpha)``.

        The genuine prongs are the deepest local minima of ``g`` (best radial
        alignment of the cross field). We take the ``expected`` smallest-``g``
        minima (merged within 10 deg). If the field is too degenerate to expose
        that many minima we fall back to an even angular spread anchored at the
        best minimum (3 prongs 120 deg apart for +1/4, 5 prongs 72 deg apart for
        -1/4), so the count invariant always holds."""
        cand = self._all_local_minima(alphas, g)  # [(angle, gval)] merged, by gval
        if expected is None or expected <= 0:
            # unknown index: keep the old threshold behaviour
            return [a for a, gv in cand if gv < 0.12]
        if len(cand) >= expected:
            return [a for a, gv in cand[:expected]]
        # degenerate: anchor at best minimum, fill with an even spread
        anchor = cand[0][0] if cand else 0.0
        return [(anchor + k * 2 * np.pi / expected) % (2 * np.pi)
                for k in range(expected)]

    @staticmethod
    def _all_local_minima(alphas, g):
        """All strict circular local minima of g, merged within 10 deg (keeping
        the deeper one), returned as (angle, gval) sorted by increasing gval."""
        n = len(g)
        mins = []
        for i in range(n):
            if g[i] <= g[(i - 1) % n] and g[i] < g[(i + 1) % n]:
                mins.append((alphas[i], g[i]))
        mins.sort(key=lambda x: x[1])  # deepest first
        merged = []
        for a, gv in mins:
            if all(abs(np.angle(np.exp(1j * (a - m)))) > np.radians(10)
                   for m, _ in merged):
                merged.append((a, gv))
        return merged

    # --- blade outlines (for "points into the blade profile?" test) ---------
    def _blade_paths(self):
        """matplotlib Paths of the inner blade loop(s), or [] if unavailable."""
        mesh = self.mesh
        loops = getattr(mesh, "blade_loops", None)
        if not loops:
            return []
        from matplotlib.path import Path as _MplPath
        return [_MplPath(np.asarray(bl, float)) for bl in loops]

    @staticmethod
    def _in_blade(p, blade_paths):
        return any(bp.contains_point((float(p[0]), float(p[1]))) for bp in blade_paths)

    def _emanate_from_outer_corners(self):
        """Experiment: emit separatrices from OUTER (corner_type==0) domain
        corners along the interior cross direction(s).

        Mirrors the blade-tip emission but for the parallelogram's outer corners,
        which normally emit nothing (valence-2 assumption). We take every cross
        arm that points into the domain (positive dot with the inward direction)
        and emit a separatrix along it, to test whether this alone splits the
        outer channels into four-sided blocks (vs. periodic tiling)."""
        seps = []
        mesh = self.mesh
        ctype = getattr(mesh, "corner_type", None)
        if ctype is None:
            return seps
        ctype = ctype.numpy() if hasattr(ctype, "numpy") else np.asarray(ctype)
        corner_ids = np.where(ctype == 0)[0].tolist()
        node_dim = mesh.x[:, 2].numpy()
        for nid in corner_ids:
            c = self.nodes[nid]
            faces = self.n2f.get(nid, [])
            if not faces:
                continue
            nbrs = set()
            for fi in faces:
                nbrs.update(self.faces[fi].tolist())
            nbrs.discard(nid)
            # Interior corner angle from the two incident boundary edges. On the
            # periodic surface an acute and an obtuse outer corner are the SAME
            # physical point (seam partners summing to 180 deg): only the obtuse
            # side carries the subdividing separatrix; the acute side's arm is
            # its one-pitch duplicate hugging the seam (and its seam crossings
            # poison the block graph). Emit only at obtuse (>90 deg) corners.
            bnbrs = [j for j in nbrs if node_dim[j] < 2]
            if len(bnbrs) >= 2:
                e = [self.nodes[j] - c for j in bnbrs[:2]]
                cosang = np.dot(e[0], e[1]) / (
                    np.linalg.norm(e[0]) * np.linalg.norm(e[1]) + 1e-18)
                if np.degrees(np.arccos(np.clip(cosang, -1, 1))) <= 95.0:
                    continue  # acute corner: no separatrix
            inward = np.mean([self.nodes[j] - c for j in nbrs], axis=0)
            ni = np.linalg.norm(inward)
            if ni < 1e-9:
                continue
            inward /= ni
            eps = 1.2 * np.mean([self._local_scale(fi) for fi in faces])
            cd = self._cross_dirs(c + eps * inward)
            if cd is None:
                continue
            # The field is wall-aligned, so at a corner the cross arms point ALONG
            # the two boundary walls. An arm along the inlet/outlet wall (t=const,
            # horizontal) integrates into a useless boundary-hugging streamline; the
            # arm that SUBDIVIDES the channel crosses it, i.e. has the largest axial
            # (t / y) component. So emit the single interior-pointing arm with the
            # largest |t-component|.
            best_d, best_dy = None, -1.0
            for al in cd:
                d = np.array([np.cos(al), np.sin(al)])
                if float(np.dot(d, inward)) <= 0.0:   # must point into the domain
                    continue
                if abs(d[1]) > best_dy:
                    best_dy, best_d = abs(d[1]), d
            if best_d is None:
                continue
            seps.append({
                "coordinates": torch.tensor(c + eps * best_d, dtype=torch.float),
                "vector": torch.tensor(best_d, dtype=torch.float),
                "singularity_coords": torch.tensor(c, dtype=torch.float),
                "face_id": -1 - nid,  # negative marker = corner origin
                "corner_origin": True,  # flag for degenerate-stub drop downstream
            })
        return seps

    def _emanate_from_blade_tips(self):
        """Emit 3 separatrices from each blade-tip (LE/TE) corner.

        Outer-domain corners (corner_type==0) emit NOTHING - their two incident
        boundary edges already form the block corner (valence 2).

        A blade tip (corner_type==1) must reach valence 5 = 2 blade boundary edges
        + 3 streamline separatrices. The 4 local cross arms are: one pointing into
        the fluid (continuing the chord, the *primary* separatrix), one pointing
        back into the blade profile (dropped), and two orthogonal to the primary.
        We pick the primary as the arm best aligned with the inward (into-fluid)
        direction, then emit it plus its two EXACT +/-90 deg rotations. Using exact
        orthogonals (not the separately-detected arms) keeps them out of the blade
        even where the cross field is noisy near the singular tip."""
        seps = []
        mesh = self.mesh
        ctype = getattr(mesh, "corner_type", None)
        if ctype is None:
            return seps
        ctype = ctype.numpy() if hasattr(ctype, "numpy") else np.asarray(ctype)
        tip_ids = np.where(ctype == 1)[0].tolist()
        # If a genuine field singularity already sits at a tip (clean/periodic
        # field), it emits its own 5 separatrices via _emanate_from_singularities;
        # the artificial 3-prong emission would then double-count the tip. Skip
        # those tips and only synthesize prongs where the field is flat.
        sing_xy = (np.array(list(mesh.singularities_coords.values()), float)
                   if getattr(mesh, "singularities_coords", None) else
                   np.zeros((0, 2)))
        tip_sing_radius = 0.08
        for nid in tip_ids:
            c = self.nodes[nid]
            if len(sing_xy) and np.min(np.linalg.norm(sing_xy - c, axis=1)) \
                    < tip_sing_radius:
                continue  # field singularity already handles this tip
            faces = self.n2f.get(nid, [])
            if not faces:
                continue
            nbrs = set()
            for fi in faces:
                nbrs.update(self.faces[fi].tolist())
            nbrs.discard(nid)
            inward = np.mean([self.nodes[j] - c for j in nbrs], axis=0)
            ni = np.linalg.norm(inward)
            if ni < 1e-9:
                continue
            inward /= ni
            eps = 1.2 * np.mean([self._local_scale(fi) for fi in faces])
            cd = self._cross_dirs(c + eps * inward)
            if cd is None:
                continue
            # primary = cross arm most aligned with the into-fluid direction
            arms = [np.array([np.cos(al), np.sin(al)]) for al in cd]
            primary = max(arms, key=lambda d: float(np.dot(d, inward)))
            ang = float(np.arctan2(primary[1], primary[0]))
            # primary + the two orthogonal directions (drop the anti-primary arm,
            # which points into the blade)
            for a in (ang, ang + np.pi / 2.0, ang - np.pi / 2.0):
                d = np.array([np.cos(a), np.sin(a)])
                seps.append({
                    "coordinates": torch.tensor(c + eps * d, dtype=torch.float),
                    "vector": torch.tensor(d, dtype=torch.float),
                    "singularity_coords": torch.tensor(c, dtype=torch.float),
                    "face_id": -1 - nid,  # negative marker = blade-tip origin
                })
        return seps
