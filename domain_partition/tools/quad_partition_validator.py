import torch
import numpy as np
from typing import Dict, List, Optional
from torch_geometric.data import Data
import networkx as nx

DEFAULT_TOLERANCES = {
    "scaled_jacobian_min": 0.3,
    "scaled_jacobian_mean": 0.85,
    "min_interior_angle": 45.0,
    "max_interior_angle": 135.0,
    "edge_length_ratio_max": 10.0,
    "singularity_efficiency_min": 0.5,
    "boundary_match_tol": 1e-3,
    "euler_tol": 0.1,
    "degenerate_angle_min": 5.0,
    "degenerate_angle_max": 175.0,
}


class QuadPartitionValidator:
    """
    Hierarchischer Pre-Filter fuer Quad-Domain-Partitionen.
    Laeuft auf der Block-Topologie (blocked_mesh) **vor** dem teuren
    Transfinite_Interpolation-Schritt.
    """

    def __init__(self,
                 blocked_mesh: Data,
                 tri_mesh: Data,
                 frame_field: Optional[torch.Tensor] = None,
                 tol_config: Optional[dict] = None,
                 strict: bool = False):
        self.blocked_mesh = blocked_mesh
        self.tri_mesh = tri_mesh
        self.frame_field = frame_field if frame_field is not None else getattr(tri_mesh, 'frame_field', None)
        self.strict = strict
        self.tol = {**DEFAULT_TOLERANCES, **(tol_config or {})}

        self._diagnostics: List[str] = []
        self._quality_score: Optional[Dict[str, float]] = None
        self._is_valid: Optional[bool] = None

        # Cache fuer haeufig genutzte Werte
        self._num_nodes = blocked_mesh.x.size(0)
        self._faces = blocked_mesh.faces  # (4, F)
        self._num_faces = self._faces.size(1)

        # Extract unique edges and boundary classification
        self._unique_edges, self._edge_counts, self._boundary_edges, self._boundary_nodes = self._extract_edge_topology()
        self._num_edges = self._unique_edges.size(0)
        self._inner_nodes = self._get_inner_nodes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_valid(self) -> bool:
        """Boolesches AND aller Hard-Checks."""
        if self._is_valid is None:
            self._run_validation()
        return self._is_valid

    def quality_score(self) -> Dict[str, float]:
        """Soft-Metriken (werden auch bei Hard-Fail berechnet)."""
        if self._quality_score is None:
            self._compute_quality_score()
        return self._quality_score

    def diagnostics(self) -> List[str]:
        """Menschenlesbare Failure-Reports."""
        if self._is_valid is None:
            self._run_validation()
        return self._diagnostics

    def passes_soft_thresholds(self, tol: Optional[dict] = None) -> bool:
        """Optionaler separater Soft-Check."""
        cfg = {**self.tol, **(tol or {})}
        qs = self.quality_score()
        ok = True
        if qs["scaled_jacobian_min"] < cfg["scaled_jacobian_min"]:
            ok = False
        if qs["scaled_jacobian_mean"] < cfg["scaled_jacobian_mean"]:
            ok = False
        if qs["min_interior_angle"] < cfg["min_interior_angle"]:
            ok = False
        if qs["max_interior_angle"] > cfg["max_interior_angle"]:
            ok = False
        if qs["edge_length_ratio_max"] > cfg["edge_length_ratio_max"]:
            ok = False
        if qs.get("singularity_efficiency", 1.0) < cfg["singularity_efficiency_min"]:
            ok = False
        return ok

    # ------------------------------------------------------------------
    # Interne Validierung
    # ------------------------------------------------------------------
    def _run_validation(self):
        self._diagnostics = []
        hard_ok = True

        # 1. Topologie
        hard_ok &= self._check_topology()
        # 2. Element-Validitaet (Inversion, degenerierte Winkel, Aspect Ratio)
        hard_ok &= self._check_element_validity()
        # 3. Boundary-Match
        hard_ok &= self._check_boundary_match()

        if not hard_ok and not self.strict:
            # Phase 1: logge Warnungen, bleibe aber gueltig
            self._diagnostics.append("WARN: Hard-Checks failed, but strict=False -> continuing")
            self._is_valid = True
        else:
            self._is_valid = hard_ok

    # ------------------------------------------------------------------
    # Hard-Checks
    # ------------------------------------------------------------------
    def _check_topology(self) -> bool:
        """Euler-Charakteristik und Valenz-Regelmaessigkeit."""
        ok = True

        # Euler-Charakteristik aus Boundary-Loops ableiten:
        # fuer k Loecher: chi = 1 - k
        # Aeussere Boundary + k innere Boundaries = k+1 Loops
        boundary_loops = self._find_boundary_loops()
        num_holes = max(0, len(boundary_loops) - 1)
        expected_euler = 1 - num_holes
        euler = self._num_nodes - self._num_edges + self._num_faces
        if abs(euler - expected_euler) > self.tol["euler_tol"]:
            self._diagnostics.append(
                f"TOPOLOGY FAIL: Euler = {euler:.2f} != {expected_euler:.2f} "
                f"(expected for {num_holes} hole(s), {len(boundary_loops)} boundary loop(s))"
            )
            ok = False
        else:
            self._diagnostics.append(
                f"Topology OK: Euler = {euler:.2f} ({num_holes} hole(s), {len(boundary_loops)} loops)"
            )

        # Valenz-Check
        degree = torch.zeros(self._num_nodes, dtype=torch.long)
        for e in self._unique_edges:
            degree[e[0]] += 1
            degree[e[1]] += 1

        # Innere Knoten muessen Valenz 4 haben
        if len(self._inner_nodes) > 0:
            inner_deg = degree[self._inner_nodes]
            bad_inner = self._inner_nodes[inner_deg != 4]
            if bad_inner.numel() > 0:
                self._diagnostics.append(
                    f"TOPOLOGY FAIL: {bad_inner.numel()} inner node(s) with valence != 4 "
                    f"(indices: {bad_inner.tolist()})"
                )
                ok = False
            else:
                self._diagnostics.append("Topology OK: All inner nodes have valence 4")

        # Boundary-Knoten muessen Valenz 2 oder 3 haben (gerade Kante vs. Ecke)
        if len(self._boundary_nodes) > 0:
            bound_deg = degree[self._boundary_nodes]
            bad_bound = self._boundary_nodes[(bound_deg < 2) | (bound_deg > 4)]
            if bad_bound.numel() > 0:
                self._diagnostics.append(
                    f"TOPOLOGY WARN: {bad_bound.numel()} boundary node(s) with unusual valence "
                    f"(indices: {bad_bound.tolist()}, degrees: {bound_deg[(bound_deg < 2) | (bound_deg > 4)].tolist()})"
                )
                # Wir behandeln das als Warnung, nicht als harten Fail,
                # weil komplexe Geometrien (NACA-Ecken) Valenz 3 haben koennen.

        return ok

    def _check_element_validity(self) -> bool:
        """Keine Inversionen, keine degenerierten Winkel, kein extremes Aspect Ratio."""
        ok = True
        nodes = self.blocked_mesh.x[:, 0:2]
        faces = self._faces

        # --- Scaled Jacobian (min pro Quad) ---
        sj = self._compute_scaled_jacobian_per_quad(nodes, faces)  # (F,)
        if (sj < 0).any():
            inv_mask = (sj < 0).nonzero(as_tuple=True)[0]
            self._diagnostics.append(
                f"ELEMENT FAIL: {inv_mask.numel()} inverted quad(s) "
                f"(scaled Jacobian < 0, indices: {inv_mask[:5].tolist()})"
            )
            ok = False
        else:
            self._diagnostics.append("Element OK: No inverted quads")

        # --- Interior Angles ---
        angles = self._compute_interior_angles(nodes, faces)  # (4, F)
        min_a = angles.min()
        max_a = angles.max()
        if min_a < self.tol["degenerate_angle_min"]:
            self._diagnostics.append(
                f"ELEMENT FAIL: Min interior angle = {min_a:.2f}deg < "
                f"{self.tol['degenerate_angle_min']}deg"
            )
            ok = False
        if max_a > self.tol["degenerate_angle_max"]:
            self._diagnostics.append(
                f"ELEMENT FAIL: Max interior angle = {max_a:.2f}deg > "
                f"{self.tol['degenerate_angle_max']}deg"
            )
            ok = False
        if ok:
            self._diagnostics.append(
                f"Element OK: Angles in [{min_a:.2f}, {max_a:.2f}] deg"
            )

        # --- Edge Length Ratio ---
        ratios = self._compute_edge_length_ratios(nodes, faces)  # (F,)
        if (ratios > self.tol["edge_length_ratio_max"]).any():
            bad = (ratios > self.tol["edge_length_ratio_max"]).nonzero(as_tuple=True)[0]
            self._diagnostics.append(
                f"ELEMENT FAIL: {bad.numel()} quad(s) with edge-length-ratio > "
                f"{self.tol['edge_length_ratio_max']} "
                f"(max ratio = {ratios.max():.2f})"
            )
            ok = False
        else:
            self._diagnostics.append(
                f"Element OK: Max edge-length-ratio = {ratios.max():.2f}"
            )

        return ok

    def _check_boundary_match(self) -> bool:
        """
        Boundary-Knoten des Tri-Meshes muessen nahe an Boundary-Knoten
        des Block-Meshes liegen.
        """
        tri = self.tri_mesh
        if not hasattr(tri, 'edge_attr') or tri.edge_attr is None:
            self._diagnostics.append("BOUNDARY SKIP: tri_mesh has no edge_attr")
            return True  # Nicht blocken, wenn Daten fehlen

        # Boundary-Koordinaten des tri_mesh
        mask_b = tri.edge_attr == 1
        if mask_b.sum() == 0:
            self._diagnostics.append("BOUNDARY SKIP: tri_mesh has no boundary edges")
            return True

        tri_b_edges = tri.edge_index[:, mask_b]
        tri_b_nodes = torch.unique(tri_b_edges)
        tri_b_coords = tri.x[tri_b_nodes, 0:2]

        # Boundary-Koordinaten des blocked_mesh
        blk_b_nodes = self._boundary_nodes
        if len(blk_b_nodes) == 0:
            self._diagnostics.append("BOUNDARY FAIL: blocked_mesh has no boundary nodes")
            return False
        blk_b_coords = self.blocked_mesh.x[blk_b_nodes, 0:2].to(tri_b_coords.dtype)

        # Pairwise nearest-neighbor distance (Hausdorff approx)
        dists = torch.cdist(tri_b_coords, blk_b_coords)  # (N_tri, N_blk)
        min_dists = dists.min(dim=1)[0]
        tol = self.tol["boundary_match_tol"]

        if (min_dists > tol).any():
            n_bad = (min_dists > tol).sum().item()
            max_bad = min_dists[min_dists > tol].max().item()
            self._diagnostics.append(
                f"BOUNDARY FAIL: {n_bad} tri-boundary node(s) > {tol} away "
                f"from nearest block-boundary node (max dist = {max_bad:.4f})"
            )
            return False

        self._diagnostics.append(
            f"Boundary OK: All tri-boundary nodes within {min_dists.max():.4f} "
            f"of block-boundary (tol={tol})"
        )
        return True

    # ------------------------------------------------------------------
    # Soft-Metriken
    # ------------------------------------------------------------------
    def _compute_quality_score(self):
        nodes = self.blocked_mesh.x[:, 0:2]
        faces = self._faces

        sj = self._compute_scaled_jacobian_per_quad(nodes, faces)
        angles = self._compute_interior_angles(nodes, faces)
        ratios = self._compute_edge_length_ratios(nodes, faces)

        self._quality_score = {
            "scaled_jacobian_min": sj.min().item(),
            "scaled_jacobian_mean": sj.mean().item(),
            "min_interior_angle": angles.min().item(),
            "max_interior_angle": angles.max().item(),
            "edge_length_ratio_max": ratios.max().item(),
            "num_quads": self._num_faces,
            "num_nodes": self._num_nodes,
            "num_edges": self._num_edges,
        }

        # Optional: Singularity Efficiency (Smin)
        sing_eff = self._compute_singularity_efficiency()
        if sing_eff is not None:
            self._quality_score["singularity_efficiency"] = sing_eff
            self._quality_score["expected_singularities"] = self._expected_singularities
            self._quality_score["actual_singularities"] = self._actual_singularities

        # Logge Soft-Metriken
        self._diagnostics.append(
            f"Quality Score: SJ_min={self._quality_score['scaled_jacobian_min']:.3f}, "
            f"SJ_mean={self._quality_score['scaled_jacobian_mean']:.3f}, "
            f"angle=[{self._quality_score['min_interior_angle']:.1f}, "
            f"{self._quality_score['max_interior_angle']:.1f}], "
            f"aspect_max={self._quality_score['edge_length_ratio_max']:.2f}"
        )
        if sing_eff is not None:
            self._diagnostics.append(
                f"Singularity Efficiency: {sing_eff:.3f} "
                f"(expected={self._expected_singularities}, actual={self._actual_singularities})"
            )

    # ------------------------------------------------------------------
    # Geometrie-Helfer (vektorisiert, PyTorch)
    # ------------------------------------------------------------------
    def _extract_edge_topology(self):
        faces = self._faces  # (4, F)
        # Alle Kanten aller Quads (gerichtet, unsortiert)
        e0 = faces[[0, 1]]
        e1 = faces[[1, 2]]
        e2 = faces[[2, 3]]
        e3 = faces[[3, 0]]
        all_edges = torch.cat([e0, e1, e2, e3], dim=1)  # (2, 4F)
        # Sortiere, damit (i,j) == (j,i)
        sorted_edges = torch.sort(all_edges, dim=0)[0].T  # (4F, 2)
        unique_edges, counts = torch.unique(sorted_edges, dim=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        boundary_nodes = torch.unique(boundary_edges) if boundary_edges.numel() > 0 else torch.tensor([], dtype=torch.long)
        return unique_edges, counts, boundary_edges, boundary_nodes

    def _get_inner_nodes(self):
        all_nodes = torch.arange(self._num_nodes, dtype=torch.long)
        boundary_set = set(self._boundary_nodes.tolist())
        inner = [n for n in all_nodes.tolist() if n not in boundary_set]
        return torch.tensor(inner, dtype=torch.long) if inner else torch.tensor([], dtype=torch.long)

    def _find_boundary_loops(self) -> List[List[int]]:
        """
        Findet geschlossene Boundary-Loops im blocked_mesh.
        Boundary-Kanten sind die, die nur zu einem Face gehoeren.
        """
        if len(self._boundary_edges) == 0:
            return []
        G = nx.Graph()
        for e in self._boundary_edges:
            G.add_edge(e[0].item(), e[1].item())
        loops = []
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            try:
                cycle = nx.find_cycle(subG)
                loops.append([u for u, v in cycle] + [cycle[-1][1]])
            except nx.NetworkXNoCycle:
                # Offene Kette -> kein Loop
                pass
        return loops

    @staticmethod
    def _compute_scaled_jacobian_per_quad(nodes, faces):
        """
        nodes: (N, 2)
        faces: (4, F)
        returns: (F,) min scaled Jacobian per quad
        """
        p = nodes[faces]  # (4, F, 2)
        jacobians = []
        for k in range(4):
            p_k = p[k]          # (F, 2)
            p_next = p[(k + 1) % 4]
            p_prev = p[(k - 1) % 4]
            a = p_next - p_k    # (F, 2)
            b = p_prev - p_k    # (F, 2)
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            norm_a = torch.norm(a, dim=1)
            norm_b = torch.norm(b, dim=1)
            sj = det / (norm_a * norm_b + 1e-12)
            jacobians.append(sj)
        jacobians = torch.stack(jacobians, dim=0)  # (4, F)
        return jacobians.min(dim=0)[0]  # (F,)

    @staticmethod
    def _compute_interior_angles(nodes, faces):
        """
        returns: (4, F) angles in degrees
        """
        p = nodes[faces]  # (4, F, 2)
        angles = []
        for k in range(4):
            p_k = p[k]
            p_prev = p[(k - 1) % 4]
            p_next = p[(k + 1) % 4]
            v1 = p_prev - p_k  # (F, 2)
            v2 = p_next - p_k  # (F, 2)
            dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            # Innenwinkel ist immer in [0, 180]
            ang = torch.atan2(torch.abs(cross), dot) * 180.0 / torch.pi
            ang = torch.clamp(ang, 0.0, 180.0)
            angles.append(ang)
        return torch.stack(angles, dim=0)  # (4, F)

    @staticmethod
    def _compute_edge_length_ratios(nodes, faces):
        """
        returns: (F,) max/min edge length per quad
        """
        p = nodes[faces]  # (4, F, 2)
        lengths = []
        for k in range(4):
            p_k = p[k]
            p_next = p[(k + 1) % 4]
            lengths.append(torch.norm(p_next - p_k, dim=1))
        lengths = torch.stack(lengths, dim=0)  # (4, F)
        return lengths.max(dim=0)[0] / (lengths.min(dim=0)[0] + 1e-12)

    # ------------------------------------------------------------------
    # Smin / Singularity Efficiency (Kowalski et al. 2015)
    # ------------------------------------------------------------------
    def _compute_singularity_efficiency(self) -> Optional[float]:
        """
        Berechnet Smin entlang der Boundary-Loops des tri_mesh.
        Returns None, wenn frame_field oder Boundary-Daten fehlen.
        """
        tri = self.tri_mesh
        if self.frame_field is None:
            return None
        if not hasattr(tri, 'edge_attr') or tri.edge_attr is None:
            return None

        mask_b = tri.edge_attr == 1
        if mask_b.sum() == 0:
            return None

        # Boundary-Graph
        b_edges = tri.edge_index[:, mask_b]
        G = nx.Graph()
        for i in range(b_edges.size(1)):
            G.add_edge(b_edges[0, i].item(), b_edges[1, i].item())

        # Finde Zyklen in jeder connected component
        cycles = []
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            try:
                cycle = nx.find_cycle(subG)
                cycles.append(cycle)
            except nx.NetworkXNoCycle:
                pass

        if len(cycles) == 0:
            return None

        # Bestimme aeusseren Zyklus (groesste Bounding-Box-Flaeche)
        def cycle_bbox_area(cycle):
            nodes_in_cycle = set()
            for u, v in cycle:
                nodes_in_cycle.add(u)
                nodes_in_cycle.add(v)
            coords = tri.x[list(nodes_in_cycle), 0:2]
            dx = coords[:, 0].max() - coords[:, 0].min()
            dy = coords[:, 1].max() - coords[:, 1].min()
            return dx * dy

        areas = [cycle_bbox_area(c) for c in cycles]
        outer_idx = int(np.argmax(areas))
        outer_cycle = cycles[outer_idx]
        inner_cycles = [c for i, c in enumerate(cycles) if i != outer_idx]

        # Winkel entlang Zyklen
        def theta_of_cycle(cycle):
            theta1 = 0.0
            theta2 = 0.0
            # cycle ist Liste von (u,v); wir gehen sequentiell
            for u, v in cycle:
                angle_u = torch.atan2(self.frame_field[u, 1], self.frame_field[u, 0]).item()
                angle_v = torch.atan2(self.frame_field[v, 1], self.frame_field[v, 0]).item()
                # Branch-alignment: Differenz modulo pi/2 in [-pi/2, pi/2]
                dphi = np.remainder(angle_v - angle_u + np.pi / 2, np.pi) - np.pi / 2
                length = torch.norm(tri.x[v, 0:2] - tri.x[u, 0:2]).item()
                if dphi > 0:
                    theta1 += dphi * length
                else:
                    theta2 += abs(dphi) * length
            return theta1, theta2

        t1_out, t2_out = theta_of_cycle(outer_cycle)
        S_outer = t1_out + t2_out
        S_inner = 0.0
        for c in inner_cycles:
            t1, t2 = theta_of_cycle(c)
            S_inner += (t1 + t2)

        Smin = abs(S_outer - S_inner)
        # Theoretisch sollte Smin nahe einer Ganzzahl sein (Anzahl Singularitaeten)
        self._expected_singularities = round(Smin)

        # Tatsaechliche Singularitaeten im blocked_mesh:
        # innere Knoten mit Valenz != 4
        degree = torch.zeros(self._num_nodes, dtype=torch.long)
        for e in self._unique_edges:
            degree[e[0]] += 1
            degree[e[1]] += 1
        if len(self._inner_nodes) > 0:
            self._actual_singularities = (degree[self._inner_nodes] != 4).sum().item()
        else:
            self._actual_singularities = 0

        if self._actual_singularities > 0:
            efficiency = self._expected_singularities / self._actual_singularities
        else:
            efficiency = 1.0 if self._expected_singularities == 0 else 0.0
        return float(efficiency)
