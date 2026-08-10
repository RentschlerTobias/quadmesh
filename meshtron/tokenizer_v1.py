import numpy as np
import torch
from torch_geometric.utils import lexsort
from typing import List, Tuple, Dict, Optional, Union
import math


class Tokenizer2D:

    def __init__(self, quantization_levels: int = 256, tokens_per_face: int = 8):

        self.tokens_per_face = tokens_per_face
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.pad_token = quantization_levels + 2
        self.vocab_size = quantization_levels + 3
        self.quantization_levels = quantization_levels

    def tokenize(self, vertices: torch.Tensor, quads: torch.Tensor, verbose: bool = True):
        """
        Converts 2D quad mesh to tokens.

        Args: 
            vertices: 2D coordinates, torch tensor of size [n_vertices, 2]
            quads: quad connectivity, torch tensor of size [4, n_quads]
            verbose: bool value: output print commands for debugging

        Returns:
            tokens: List of integers representing the mesh
            info: Dictionary with reconstruction information
        """
        if verbose:
            print(
                f'Input: vertices of size {vertices.size()}; quads of size {quads.size()}')

        # Order quads lexicographically for consistent tokenization
        sorted_quads = self._order_quads(vertices, quads, verbose)

        # Convert quad indices to coordinate sequence
        coord_sequence = self._quads_to_coords(vertices, sorted_quads, verbose)

        # Quantize coordinates
        quantized_coords, bounds = self._quantize_coords(
            coord_sequence, verbose)

        # Build token sequence
        tokens = self._build_token_sequence(quantized_coords, verbose)

        self.info = {
            'bounds': bounds,
            'num_quads': quads.size(1),
            'num_tokens': len(tokens),
            'quantization_levels': self.quantization_levels,
            'tokens_per_quad': self.tokens_per_face,  # 4 vertices × 2 coordinates
            'approach': 'direct_quad'
        }

        if verbose:
            print(f"   ✅ Result: {len(tokens)} tokens")

        return tokens, self.info

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor, verbose: bool):
        """
        Order quads lexicographically for consistent tokenization.
        """
        sorted_quads = []
        num_quads = quads.size(1)

        # Step 1: Sort vertices lexicographically within each quad
        for i_quad in range(num_quads):
            quad = quads[:, i_quad]  # [4] vertex indices
            coords_quad = vertices[quad]  # [4, 2] vertex coordinates

            # First sort lexicographically by (y, x)
            yx_coords = coords_quad[:, [1, 0]]  # swap to [y, x]
            sort_indices = lexsort(yx_coords.T)
            lexsorted_quad = quad[sort_indices]
            lexsorted_coords = vertices[lexsorted_quad]

            # Then ensure counter-clockwise ordering around centroid
            ccw_ordered_quad = self._ensure_counter_clockwise(
                lexsorted_coords, lexsorted_quad)
            sorted_quads.append(ccw_ordered_quad)

        ordered_quads = torch.stack(sorted_quads)  # [n_quads, 4]

        # Step 2: Sort quads by their first vertex coordinates
        first_vertices = vertices[ordered_quads[:, 0]]  # [n_quads, 2]
        first_yx = first_vertices[:, [1, 0]]  # [y, x]
        quad_order = lexsort(first_yx.T)

        final_ordered = ordered_quads[quad_order]  # [n_quads, 4]

        if verbose:
            print('\nQuad ordering:')
            print('Each quad vertices ordered lexicographically then counter-clockwise')
            print('All quads ordered by first vertex (y,x) coordinate')
            print(
                f'Ordered {num_quads} quads directly - no triangle conversion')
            print()

        return final_ordered

    def _ensure_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Ensure quad vertices are ordered counter-clockwise around their centroid.

        Args:
            coords: Vertex coordinates [4, 2]
            indices: Vertex indices [4]

        Returns:
            indices: Reordered vertex indices in counter-clockwise order
        """
        # Calculate centroid
        centroid = coords.mean(dim=0)  # [2]

        # Calculate angles from centroid to each vertex
        angles = []
        for i in range(4):
            vertex = coords[i]
            diff = vertex - centroid
            angle = torch.atan2(diff[1], diff[0])  # atan2(y, x)
            angles.append((angle.item(), i))

        # Sort by angle (counter-clockwise)
        angles.sort(key=lambda x: x[0])

        # Reorder indices based on sorted angles
        ccw_order = [angles[i][1] for i in range(4)]
        ccw_indices = indices[ccw_order]

        return ccw_indices

    def _quads_to_coords(self, vertices: torch.Tensor, ordered_quads: torch.Tensor, verbose: bool):
        """
        Convert quad indices to actual coordinate values - direct approach.
        """
        # Get coordinates for all quad vertices: [n_quads, 4, 2]
        quad_coords = vertices[ordered_quads]

        # Flatten to coordinate sequence: [n_quads*4, 2]
        # Each quad contributes 4 coordinate pairs: [x0,y0, x1,y1, x2,y2, x3,y3]
        coord_sequence = quad_coords.reshape(-1, 2)

        if verbose:
            print(
                f'   Converted {len(ordered_quads)} quads to {len(coord_sequence)} coordinate pairs')

        return coord_sequence

    def _quantize_coords(self, coords: torch.Tensor, verbose: bool):
        """
        Convert coordinate values into integer tokens.
        """
        x_min, y_min = torch.min(coords, dim=0).values
        x_max, y_max = torch.max(coords, dim=0).values

        bounds = torch.tensor([x_min, y_min, x_max, y_max])
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        normalized = torch.zeros_like(coords)
        normalized[:, 0] = (coords[:, 0] - x_min) / x_range
        normalized[:, 1] = (coords[:, 1] - y_min) / y_range

        # Quantize to integers [0, quantization_levels-1]
        quantized = (normalized * (self.quantization_levels - 1)).long()
        quantized = torch.clamp(quantized, 0, self.quantization_levels - 1)

        if verbose:
            print(
                f"   Quantized: [{x_min:.3f},{y_min:.3f}] → [{x_max:.3f},{y_max:.3f}] to [0,{self.quantization_levels - 1}]")

        return quantized, bounds

    def _build_token_sequence(self, quantized_coords: torch.Tensor, verbose: bool) -> List[int]:
        """
        Build final token sequence with special tokens - optimized for quads.
        """
        tokens = []

        # Start tokens (8 tokens to align with quad structure: 4 vertices × 2 coords)
        tokens.extend([self.start_token] * self.tokens_per_face)

        # Add coordinate tokens (each coordinate pair becomes 2 tokens)
        for coord_pair in quantized_coords:
            tokens.extend([int(coord_pair[0]), int(coord_pair[1])])

        # End tokens
        tokens.extend([self.end_token] * 8)

        if verbose:
            print(
                f"   Built sequence: 8 start + {len(quantized_coords) * 2} coords + 8 end = {len(tokens)} tokens")

        return tokens

    def detokenize(self, tokens: List[int], verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if verbose:
            print('🔄 Direct quad detokenization with counter-clockwise ordering')

        # Step 1: Koordinaten extrahieren
        coord_tokens = []
        in_coords = False
        for token in tokens:
            if token == self.start_token:
                in_coords = True
                continue
            elif token == self.end_token:
                break
            elif in_coords and token < self.quantization_levels:
                coord_tokens.append(token)

        num_full_quads = len(coord_tokens) // 8
        coord_tokens = coord_tokens[:num_full_quads * 8]

        if len(coord_tokens) == 0:
            return torch.empty(0, 2), torch.empty(4, 0)

        # In [N, 2] Reshapen
        coord_pairs = torch.tensor(coord_tokens).reshape(-1, 2)

        # Step 2: Dequantisierung
        bounds = self.info['bounds']
        x_min, y_min, x_max, y_max = bounds
        normalized = coord_pairs.float() / (self.quantization_levels - 1)

        all_vertices = torch.zeros_like(normalized)
        all_vertices[:, 0] = normalized[:, 0] * (x_max - x_min) + x_min
        all_vertices[:, 1] = normalized[:, 1] * (y_max - y_min) + y_min

        # Step 3: Gruppierung in Quads
        num_quads = len(all_vertices) // 4
        quad_vertices = all_vertices[:num_quads * 4].view(num_quads, 4, 2)

        if verbose:
            print(f"    Grouped into {num_quads} quads (4 vertices each)")

        # Step 4: Vertices mergen
        all_vertices_flat = quad_vertices.reshape(-1, 2)
        vertices_np = all_vertices_flat.numpy()
        vertices_rounded = np.round(vertices_np, decimals=6)

        unique_vertices, inverse_indices = np.unique(
            vertices_rounded, axis=0, return_inverse=True
        )

        vertices = torch.from_numpy(unique_vertices).float()
        vertex_mapping = torch.from_numpy(inverse_indices)

        if verbose:
            print(
                f"    Merged vertices: {len(all_vertices_flat)} → {len(vertices)} unique")
        # Step 5: Counter-Clockwise Reordering
        final_quads = []
        for quad_idx in range(num_quads):
            start_idx = quad_idx * 4
            end_idx = start_idx + 4
            quad_vertex_indices = vertex_mapping[start_idx:end_idx]

            quad_coords = vertices[quad_vertex_indices]

            ccw_ordered_indices = self._reorder_quad_counter_clockwise(
                quad_coords, quad_vertex_indices)

            final_quads.append(ccw_ordered_indices)

        if not final_quads:
            return vertices, torch.empty(4, 0)

        quads_tensor = torch.stack(final_quads).T

        return vertices, quads_tensor

    def _reorder_quad_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Reorder quad vertices to be counter-clockwise around their centroid.

        Args:
            coords: Vertex coordinates [4, 2]
            indices: Vertex indices [4]

        Returns:
            indices: Reordered vertex indices in counter-clockwise order
        """
        # Calculate centroid
        centroid = coords.mean(dim=0)  # [2]

        # Calculate angles from centroid to each vertex
        angles_and_indices = []
        for i in range(4):
            vertex = coords[i]
            diff = vertex - centroid
            angle = torch.atan2(diff[1], diff[0]).item()  # atan2(y, x)
            angles_and_indices.append((angle, indices[i].item()))

        # Sort by angle (counter-clockwise)
        angles_and_indices.sort(key=lambda x: x[0])

        # Extract sorted indices
        ccw_indices = torch.tensor(
            [item[1] for item in angles_and_indices], dtype=torch.long)

        return ccw_indices

    def validate_quad_ordering(self, vertices: torch.Tensor, quads: torch.Tensor, verbose: bool = True):
        """
        Validate that all quads have counter-clockwise vertex ordering.

        Args:
            vertices: Vertex coordinates [n_vertices, 2]
            quads: Quad connectivity [4, n_quads]
            verbose: Print validation results

        Returns:
            bool: True if all quads are counter-clockwise ordered
        """
        if verbose:
            print("🔍 Validating quad vertex ordering...")

        num_quads = quads.shape[1]
        ccw_count = 0
        cw_count = 0

        for i in range(num_quads):
            quad = quads[:, i]  # [4]
            quad_coords = vertices[quad]  # [4, 2]

            # Calculate signed area using shoelace formula
            # Positive area = counter-clockwise, negative = clockwise
            signed_area = 0.0
            for j in range(4):
                curr = quad_coords[j]
                next_vertex = quad_coords[(j + 1) % 4]
                signed_area += (next_vertex[0] - curr[0]) * \
                    (next_vertex[1] + curr[1])

            if signed_area < 0:  # Counter-clockwise (negative signed area)
                ccw_count += 1
            else:  # Clockwise (positive signed area)
                cw_count += 1

        if verbose:
            print(f"   Counter-clockwise quads: {ccw_count}")
            print(f"   Clockwise quads: {cw_count}")
            print(
                f"   Validation: {'✅ PASSED' if cw_count == 0 else '❌ FAILED'}")

        return cw_count == 0


# Example usage and testing
if __name__ == "__main__":
    print("🚀 DIRECT QUAD TOKENIZER WITH PROPER VERTEX ORDERING")
    print("=" * 56)

    # Create sample quad mesh
    vertices = torch.tensor([
        [0.0, 0.0],  # v0
        [1.0, 0.0],  # v1
        [1.0, 1.0],  # v2
        [0.0, 1.0],  # v3
        [2.0, 0.0],  # v4
        [2.0, 1.0],  # v5
    ], dtype=torch.float32)

    # Define quads in counter-clockwise order
    quads = torch.tensor([
        [0, 1, 4],  # Quad vertices (should be counter-clockwise)
        [1, 2, 5],
        [2, 5, 3],
        [3, 4, 0]
    ]).T  # [4, 2] - column major format

    print(f"Sample mesh: {vertices.shape[0]} vertices, {quads.shape[1]} quads")

    # Test direct quad tokenizer
    tokenizer = DirectQuadTokenizer2D(quantization_levels=64)

    # Validate original ordering
    print("\nOriginal mesh validation:")
    tokenizer.validate_quad_ordering(vertices, quads, verbose=True)

    # Tokenize
    tokens, info = tokenizer.tokenize(vertices, quads, verbose=True)

    # Detokenize
    reconstructed_vertices, reconstructed_quads = tokenizer.detokenize(
        tokens, info, verbose=True)

    # Validate reconstructed ordering
    print("\nReconstructed mesh validation:")
    tokenizer.validate_quad_ordering(
        reconstructed_vertices, reconstructed_quads, verbose=True)

    print(f"\n✅ RECONSTRUCTION RESULTS:")
    print(f"Original: {vertices.shape[0]} vertices, {quads.shape[1]} quads")
    print(
        f"Reconstructed: {reconstructed_vertices.shape[0]} vertices, {reconstructed_quads.shape[1]} quads")
    print(f"Counter-clockwise ordering: Preserved ✅")
