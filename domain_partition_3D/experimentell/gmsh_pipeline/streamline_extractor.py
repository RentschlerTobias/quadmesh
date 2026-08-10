import openmesh as om
import numpy as np


class StreamlineExtractor:
    def __init__(self, vertices, faces):
        self.mesh = om.PolyMesh()
        # vertices shape: (N, 3), faces shape: (M, 4)
        self.vh = [self.mesh.add_vertex(v) for v in vertices]

        for f in faces:
            self.mesh.add_face([self.vh[i] for i in f])

    def get_streamlines(self, decimals=9):
        streamlines = []
        # We must track visited HALF-EDGES to avoid re-tracing segments
        visited_he = set()

        # 1. Internal Singularities (Valence != 4)
        self.singularities = [v for v in self.mesh.vertices()
                              if not self.mesh.is_boundary(v) and self.mesh.valence(v) != 4]

        for v in self.singularities:
            for voh in self.mesh.voh(v):
                streamlines.append(self._trace(v, voh))
                # streamlines.append(self._trace(voh))

        # 2. Boundaries
        for heh in self.mesh.halfedges():
            # Check if it's a boundary half-edge and not yet processed
            if self.mesh.is_boundary(heh) and heh.idx() not in visited_he:
                streamlines.append(self._trace_boundary(heh, visited_he))

        return self._filter_unique(streamlines, decimals)

    def _trace(self, singularity, start_he, max_steps=10000):
        """
        Traces a path by choosing the outgoing edge with the smallest 
        angular deviation at every single node.
        """
        # Get start point (Singularity)
        p_prev = np.array(self.mesh.point(singularity))
        path = [p_prev]

        curr = start_he
        local_visited = {curr.idx()}

        for i in range(max_steps):
            # Target of current half-edge (P_curr)
            to_v = self.mesh.to_vertex_handle(curr)
            p_curr = np.array(self.mesh.point(to_v))
            path.append(p_curr)

            # Stop if boundary or singularity is reached
            if self.mesh.is_boundary(to_v) or self.mesh.valence(to_v) != 4:
                break

            # Calculate the incoming direction vector
            v_in = p_curr - p_prev
            v_in_norm = np.linalg.norm(v_in)
            if v_in_norm < 1e-10:
                break
            v_in /= v_in_norm

            best_he = None
            max_dot = -1.0

            # Inspect all outgoing half-edges at the current node
            for voh in self.mesh.voh(to_v):
                p_next = np.array(self.mesh.point(
                    self.mesh.to_vertex_handle(voh)))
                v_out = p_next - p_curr

                v_out_norm = np.linalg.norm(v_out)
                if v_out_norm < 1e-10:
                    continue
                v_out /= v_out_norm

                # Dot product: higher means the vectors are more aligned (1.0 = 0° deviation)
                dot = np.dot(v_in, v_out)
                if dot > max_dot:
                    max_dot = dot
                    best_he = voh

            # Check if the best match is still a valid continuation
            # (Stop if the path would take a sharp turn > 45 degrees)
            if best_he is None or max_dot < 0.7:
                break

            # Update for the next step
            p_prev = p_curr
            curr = best_he

            # Detect cycles
            if curr.idx() in local_visited:
                break
            local_visited.add(curr.idx())

        return np.array(path)

    def _trace_boundary(self, start_he, visited):
        """Trace the perimeter of a surface."""
        path = []
        curr = start_he
        while curr.idx() not in visited:
            visited.add(curr.idx())
            path.append(self.mesh.point(self.mesh.from_vertex_handle(curr)))

            # Move to next boundary half-edge
            curr = self.mesh.next_halfedge_handle(curr)
            if curr == start_he:
                break

        # Add final point to close the loop or finish the line
        path.append(self.mesh.point(self.mesh.from_vertex_handle(curr)))
        return np.array(path)

    def _filter_unique(self, lines, decimals):
        """Standard filtering logic."""
        unique, seen = [], set()
        for l in lines:
            if len(l) < 2:
                continue
            p = np.round(l, decimals)
            rev = p[::-1]
            # Use lexsort to pick a consistent direction (canonical form)
            target = rev if np.array_repr(rev) < np.array_repr(p) else p
            h = target.tobytes()
            if h not in seen:
                seen.add(h)
                unique.append(l)
        return unique
