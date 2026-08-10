import numpy as np
import torch
import openmesh as om
from torch_geometric.utils import lexsort
from typing import List, Tuple, Dict, Optional, Union
import math


class Tokenizer2D:
    def __init__(self, quantization_levels, verbose=False, max_length_padding: Optional[int] = None,
                 n_start_end_tokens_repeat: Optional[int] = 8, sorting_strategy: str = 'topological'):

        self.verbose = verbose
        # 'lexicographical' oder 'topological'
        self.sorting_strategy = sorting_strategy
        self.tokens_per_face = 8
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.pad_token = quantization_levels + 2
        self.vocab_size = quantization_levels + 3
        self.quantization_levels = quantization_levels
        self.n_start_end_tokens_repeat = n_start_end_tokens_repeat
        self.bounds = None
        self.max_length_token_sequence = 0.0
        self.min_length_token_sequence = float("inf")

        # Padding Logic (unverändert)
        if max_length_padding is not None:
            if max_length_padding % self.tokens_per_face != 0:
                self.max_length_padding = max_length_padding - \
                    (max_length_padding % self.tokens_per_face)
            else:
                self.max_length_padding = max_length_padding
        else:
            self.max_length_padding = None

    def _torch_to_openmesh(self, vertices: torch.Tensor, quads: torch.Tensor):
        """ Hilfsmethode zur Konvertierung für die topologische Sortierung. """
        mesh = om.PolyMesh()
        v_np = vertices.detach().cpu().numpy()
        f_np = quads.detach().cpu().numpy().T

        # OpenMesh benötigt 3D Punkte
        vh_list = [mesh.add_vertex(np.array([v[0], v[1], 0.0])) for v in v_np]
        for f_idx in f_np:
            mesh.add_face([vh_list[i] for i in f_idx])
        return mesh

    def _get_topological_order(self, vertices: torch.Tensor, quads: torch.Tensor):
        """ Sortiert Quads basierend auf der Halfedge-Konnektivität (DFS). """
        mesh = self._torch_to_openmesh(vertices, quads)
        n_faces = mesh.n_faces()
        visited = np.zeros(n_faces, dtype=bool)
        sorted_face_indices = []

        # Startpunkt finden (Inlet / am weitesten links)
        min_x = float('inf')
        start_handle = mesh.face_handle(0)
        for fh in mesh.faces():
            center_x = np.mean([mesh.point(vh)
                               for vh in mesh.fv(fh)], axis=0)[0]
            if center_x < min_x:
                min_x = center_x
                start_handle = fh

        # DFS Loop (um alle Inseln zu erwischen)
        for i in range(n_faces):
            root_fh = start_handle if i == 0 else mesh.face_handle(i)
            if not visited[root_fh.idx()]:
                stack = [root_fh]
                while stack:
                    curr_fh = stack.pop()
                    idx = curr_fh.idx()
                    if visited[idx]:
                        continue
                    visited[idx] = True
                    sorted_face_indices.append(idx)

                    # Nutze ff() Circulator aus der OpenMesh Doku
                    for neighbor in mesh.ff(curr_fh):
                        if not visited[neighbor.idx()]:
                            stack.append(neighbor)

        return torch.tensor(sorted_face_indices, dtype=torch.long)

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor):
        """ 
        Wählt die Sortierstrategie basierend auf self.sorting_strategy.
        """
        if self.verbose:
            print(f"Applying sorting strategy: {self.sorting_strategy}")

        if self.sorting_strategy == 'topological':
            # 1. Topologische Reihenfolge bestimmen
            topo_order = self._get_topological_order(vertices, quads)

            # 2. Innerhalb der Quads CCW sicherstellen (wie zuvor)
            sorted_quads = []
            for i in topo_order:
                quad = quads[:, i]
                coords_quad = vertices[quad]
                ccw_ordered_quad = self._ensure_counter_clockwise(
                    coords_quad, quad)
                sorted_quads.append(ccw_ordered_quad)

            return torch.stack(sorted_quads)

        else:  # Default: 'lexicographical' (dein ursprünglicher Code)
            sorted_quads = []
            num_quads = quads.size(1)
            for i_quad in range(num_quads):
                quad = quads[:, i_quad]
                coords_quad = vertices[quad]
                xy_coords = coords_quad[:, [0, 1]]
                sort_indices = lexsort(xy_coords.T)
                lexsorted_quad = quad[sort_indices]
                lexsorted_coords = vertices[lexsorted_quad]
                ccw_ordered_quad = self._ensure_counter_clockwise(
                    lexsorted_coords, lexsorted_quad)
                sorted_quads.append(ccw_ordered_quad)

            ordered_quads = torch.stack(sorted_quads)
            first_vertices = vertices[ordered_quads[:, 0]]
            quad_order = lexsort(first_vertices.T)
            return ordered_quads[quad_order]

    # ... (Rest der Methoden wie tokenize, _ensure_counter_clockwise, _quantize_coords, detokenize bleiben identisch)
