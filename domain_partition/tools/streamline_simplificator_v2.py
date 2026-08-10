from scipy.interpolate import splprep, splev
import numpy as np
from collections import defaultdict
import networkx as nx
import torch
from sklearn.cluster import DBSCAN
from torch._prims_common import dtype_to_type
from torch_geometric.data import Data


class StreamlineSimplificator_v2:
    def __init__(self, mesh):

        self.mesh = mesh

        # Pre-process streamlines
        Singularity, Streamlines = self.pre_processing(mesh.streamlines)
        self.Singularity, self.Streamlines = self.cut_streamlines(Singularity, Streamlines)
        self.mesh.streamlines = self.merge_streamlines(self.Singularity, self.Streamlines)

        # Generate splines
        self.streamline_splines = self.get_streamlines_as_splines()

        # Find intersections
        print('Searching for intersections...')
        self.intersection_data, self.intersections = self.find_all_intersections()

        # Store intersection data
        self.mesh.streamline_intersections = self.intersection_data
        self.mesh.streamline_intersections_points = self.intersections

        # Generate quad edges and mesh
        self.quad_edges = self.split_splines_at_intersections(self.intersection_data)
        self.edges_subdomain, self.nodes_subdomain, self.edge_points = self.extract_subdomain_arrays(
            self.intersection_data)

        # Generate quad mesh with improved face detection
        self.quad_mesh = self.get_mesh_improved()
        self.quad_mesh.streamlines = self.reconstruct_streamlines_from_edges(self.quad_edges)

        # Add graph attributes
        self.add_graph_attr()

    def get_mesh_improved(self):
        """Improved mesh generation with better face detection"""
        nodes = torch.tensor(self.nodes_subdomain, dtype=torch.float32)

        # Generate faces with improved algorithm
        temporary_faces = self.get_faces_improved()
        faces = self.delete_invalid_faces(temporary_faces, nodes)

        # Create edge index
        edge_index = self.create_edge_index_from_faces(faces)
        mesh = Data(x=nodes, edge_index=edge_index, faces=faces)

        return mesh

    def create_edge_index_from_faces(self, faces):
        """Create edge index from quad faces"""
        if faces.size(1) == 0:
            return torch.tensor([[], []], dtype=torch.long)

        edges = []
        for i in range(faces.size(1)):
            face = faces[:, i]
            # Add edges for each quad face
            for j in range(4):
                edges.append([face[j], face[(j + 1) % 4]])
                edges.append([face[(j + 1) % 4], face[j]])  # Bidirectional

        edge_index = torch.tensor(edges, dtype=torch.long).T
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index

    def get_faces_improved(self):
        """Improved face generation handling airfoil boundaries"""
        num_edges = self.edges_subdomain[0].size
        edges = []
        edge_centers = {}  # Store edge centers for disambiguation

        # Build edge list with center points
        for i in range(num_edges):
            edge = (self.edges_subdomain[0][i], self.edges_subdomain[1][i])
            edges.append(edge)

            # Calculate edge center for disambiguation
            if len(self.edge_points[i]) > 0:
                edge_center = np.mean(self.edge_points[i], axis=0)
                edge_centers[edge] = edge_center
                edge_centers[(edge[1], edge[0])] = edge_center  # Reverse edge

        # Create graph
        graph = nx.Graph()
        graph.add_edges_from(edges)

        # Find all simple cycles (potential faces)
        faces = list(nx.simple_cycles(graph, length_bound=4))

        quad_faces = []
        processed_regions = set()

        for face in faces:
            if len(face) == 4:
                # Sort face vertices to create a unique identifier
                face_id = tuple(sorted(face))

                if face_id not in processed_regions:
                    # Validate face
                    if self.is_valid_quad_face(face, edge_centers):
                        quad_faces.append(face)
                        processed_regions.add(face_id)

        if len(quad_faces) == 0:
            print('Warning: No valid quad faces found')
            return torch.tensor([], dtype=torch.long).reshape(4, 0)

        return torch.tensor(quad_faces, dtype=torch.long).T

    def is_valid_quad_face(self, face, edge_centers):
        """Check if a quad face is valid (handles airfoil boundaries)"""
        if len(face) != 4:
            return False

        # Check if all edges exist
        for i in range(4):
            edge = (face[i], face[(i + 1) % 4])
            if edge not in edge_centers and (edge[1], edge[0]) not in edge_centers:
                return False

        # Calculate face center
        face_center = np.zeros(2)
        for vertex_idx in face:
            if vertex_idx < len(self.nodes_subdomain):
                face_center += self.nodes_subdomain[vertex_idx]
        face_center /= 4

        # Check if face is inside the domain (not crossing airfoil)
        if hasattr(self.mesh, 'airfoil_boundary'):
            if self.is_face_crossing_airfoil(face, face_center):
                return False

        # Check face convexity
        return self.is_convex_quad(face)

    def is_convex_quad(self, face):
        """Check if a quad is convex"""
        if len(face) != 4:
            return False

        vertices = []
        for idx in face:
            if idx < len(self.nodes_subdomain):
                vertices.append(self.nodes_subdomain[idx])

        if len(vertices) != 4:
            return False

        # Check cross products for convexity
        for i in range(4):
            v1 = vertices[(i + 1) % 4] - vertices[i]
            v2 = vertices[(i + 2) % 4] - vertices[(i + 1) % 4]
            cross = v1[0] * v2[1] - v1[1] * v2[0]

            if i == 0:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False  # Not convex

        return True

    def is_face_crossing_airfoil(self, face, face_center):
        """Check if face crosses the airfoil boundary"""
        # This is a placeholder - implement based on your airfoil geometry
        # You might check if edges cross the airfoil or if the face center
        # is inside the airfoil
        return False

    def find_all_intersections(self, tolerance=1e-6):
        """Find all intersections between streamlines with improved handling"""
        splines = self.streamline_splines
        all_intersections = []

        for i in range(len(splines)):
            for j in range(i + 1, len(splines)):
                if splines[i] is None or splines[j] is None:
                    continue

                intersections = self.find_spline_intersections_improved(
                    splines[i], splines[j], tolerance)

                for point, t1, t2 in intersections:
                    all_intersections.append((point, i, t1, j, t2))

        # Process and deduplicate intersections
        unique_points = []
        spline_intersections = defaultdict(list)
        connectivity = defaultdict(list)

        for point, spline1_idx, t1, spline2_idx, t2 in all_intersections:
            # Find or create point index
            point_idx = None
            for idx, existing_point in enumerate(unique_points):
                if np.linalg.norm(np.array(point) - np.array(existing_point)) < tolerance:
                    point_idx = idx
                    break

            if point_idx is None:
                point_idx = len(unique_points)
                unique_points.append(point)

            # Update data structures
            spline_intersections[spline1_idx].append((point_idx, t1))
            spline_intersections[spline2_idx].append((point_idx, t2))

            if spline1_idx not in connectivity[point_idx]:
                connectivity[point_idx].append(spline1_idx)
            if spline2_idx not in connectivity[point_idx]:
                connectivity[point_idx].append(spline2_idx)

        # Sort by parameter value
        for spline_idx in spline_intersections:
            spline_intersections[spline_idx].sort(key=lambda x: x[1])

        intersection_coords = [torch.tensor(p, dtype=torch.float32) for p in unique_points]

        return {
            'points': unique_points,
            'spline_intersections': dict(spline_intersections),
            'connectivity': dict(connectivity)
        }, intersection_coords

    def find_spline_intersections_improved(self, spline1, spline2, tolerance=1e-5, num_samples=100):
        """Improved intersection finding with better numerical stability"""
        tck1, u1 = spline1
        tck2, u2 = spline2

        # Sample points along splines
        u1_fine = np.linspace(0, 1, num_samples)
        u2_fine = np.linspace(0, 1, num_samples)

        points1 = np.array(splev(u1_fine, tck1)).T
        points2 = np.array(splev(u2_fine, tck2)).T

        # Find potential intersections using proximity
        potential_intersections = []

        for i in range(len(points1) - 1):
            for j in range(len(points2) - 1):
                # Check segment proximity
                seg1_start, seg1_end = points1[i], points1[i + 1]
                seg2_start, seg2_end = points2[j], points2[j + 1]

                # Quick bounding box check
                if self.segments_may_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
                    u1_val = u1_fine[i] + (u1_fine[i + 1] - u1_fine[i]) / 2
                    u2_val = u2_fine[j] + (u2_fine[j + 1] - u2_fine[j]) / 2
                    potential_intersections.append((u1_val, u2_val))

        # Refine intersections
        confirmed_intersections = []

        for u1_val, u2_val in potential_intersections:
            try:
                from scipy.optimize import minimize

                # Define distance function
                def distance_squared(params):
                    t1, t2 = params
                    point1 = np.array(splev(t1, tck1)).reshape(2)
                    point2 = np.array(splev(t2, tck2)).reshape(2)
                    return np.sum((point1 - point2) ** 2)

                # Optimize to find exact intersection
                result = minimize(distance_squared, [u1_val, u2_val],
                                  bounds=[(0, 1), (0, 1)],
                                  method='L-BFGS-B')

                if result.success and result.fun < tolerance ** 2:
                    t1_intersect, t2_intersect = result.x
                    point1 = np.array(splev(t1_intersect, tck1)).reshape(2)
                    point2 = np.array(splev(t2_intersect, tck2)).reshape(2)
                    intersection_point = (point1 + point2) / 2

                    # Check for duplicates
                    is_duplicate = False
                    for existing_point, _, _ in confirmed_intersections:
                        if np.linalg.norm(np.array(existing_point) - intersection_point) < tolerance:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        confirmed_intersections.append(
                            (tuple(intersection_point), t1_intersect, t2_intersect))
            except:
                continue

        return confirmed_intersections

    def segments_may_intersect(self, p1, p2, p3, p4):
        """Check if two line segments may intersect using bounding boxes"""
        offset = 0.05

        min_x1, max_x1 = min(p1[0], p2[0]) - offset, max(p1[0], p2[0]) + offset
        min_y1, max_y1 = min(p1[1], p2[1]) - offset, max(p1[1], p2[1]) + offset
        min_x2, max_x2 = min(p3[0], p4[0]) - offset, max(p3[0], p4[0]) + offset
        min_y2, max_y2 = min(p3[1], p4[1]) - offset, max(p3[1], p4[1]) + offset

        return (min_x1 <= max_x2 and max_x1 >= min_x2 and
                min_y1 <= max_y2 and max_y1 >= min_y2)

    def cut_streamlines(self, Singularity, Streamlines):
        print("\n function cut_streamlines \n")
        for key in Singularity.keys():
            streamlines_ending = Singularity[key]["ending_streamlines"]

            if len(streamlines_ending) == 0:  # Fixed: was <0
                continue

            for streamline_ends_here in streamlines_ending:
                if streamline_ends_here not in Streamlines:
                    continue

                smallest_distance = np.inf
                starting_singularity = Streamlines[streamline_ends_here]["starting_singularity"]

                if starting_singularity not in Singularity:
                    continue

                singularity_coords = Singularity[key]["coords"]
                starting_streamlines = Singularity[starting_singularity]["starting_streamlines"]

                for starting_streamline in starting_streamlines:
                    if starting_streamline not in Streamlines:
                        continue

                    coords = torch.from_numpy(Streamlines[starting_streamline]["coords"])
                    if coords is None or len(coords) == 0:
                        continue

                    distance = torch.linalg.norm(coords - singularity_coords, axis=1)
                    distance_min_idx = torch.argmin(distance)
                    distance_min = distance[distance_min_idx]

                    if distance_min < smallest_distance:
                        smallest_distance = distance_min
                        streamline_to_cut = starting_streamline
                        idx_to_cut = distance_min_idx
                        sing_to_merge = key

                if 'streamline_to_cut' in locals() and Streamlines[streamline_to_cut]["ending_singularity"] != sing_to_merge:
                    streamline_coords = (Streamlines[streamline_to_cut]["coords"])
                    sing_coords = (Singularity[sing_to_merge]["coords"])
                    cutted_streamline = streamline_coords[0:idx_to_cut, :]
                    new_streamline = np.array(torch.cat((cutted_streamline, sing_coords), 0))

                    # Streamlines[streamline_to_cut]["coords"] = new_streamline
                    Streamlines[cutted_streamline]["coords"] = new_streamline
                    Streamlines[streamline_to_cut]["ending_singularity"] = sing_to_merge
                    print('Streamline cutted in function cut_streamlines')

        return Singularity, Streamlines

    def pre_processing(self, streamlines):

        print("\n function pre_processing \n")
        tol = 10e-3
        tol_big = 25e-3
        Singularity = {}
        Streamlines = {}

        mask_c0_nodes = self.mesh.x[:, 2] == 0
        c0_nodes = self.mesh.x[mask_c0_nodes, 0:2]
        singularity_coords  = torch.tensor([self.mesh.singularities_coords[sing] for sing in self.mesh.singularities_coords])

        streamline_termination_nodes = torch.cat((c0_nodes, singularity_coords), 0)

        for j in range(streamline_termination_nodes.size(0)):
            Singularity[j] = {"ending_streamlines": [], "starting_streamlines": [], "coords": streamline_termination_nodes[j], "is_boundary": j >= len(singularity_coords)}

        for i in range(len(streamlines)):
            Streamlines[i] = {"ending_singularity": None, "starting_singularity": None, "coords": streamlines[i], "starts_at_boundary": False, "ends_at_boundary": False}

        for i in range(len(streamlines)):
            streamline = torch.from_numpy(streamlines[i])
            start = streamline[0]
            end = streamline[-1]

            for j in range(streamline_termination_nodes.size(0)):
                termination_node = streamline_termination_nodes[j, :]
                distance_start = torch.linalg.norm(start - termination_node)
                distance_end = torch.linalg.norm(end - termination_node)

                if distance_start < tol:
                    Singularity[j]["starting_streamlines"].append(i)

                    Streamlines[i]["starting_singularity"] = j
                    Streamlines[i]["starts_at_boundary"] = Singularity[j]["is_boundary"]
                elif distance_end < tol:
                    Singularity[j]["ending_streamlines"].append(i)

                    Streamlines[i]["ending_singularity"] = j
                    Streamlines[i]["ends_at_boundary"] = Singularity[j]["is_boundary"]
                else:

                    distance = torch.linalg.norm(streamline - termination_node, axis=1)
                    distance_min_idx = torch.argmin(distance)
                    distance_min = distance[distance_min_idx]

                    if distance_min < tol:

                        cutted_streamline = streamline[0:distance_min_idx, :]

                        print(cutted_streamline.size())

                        print(termination_node.size())

                        Singularity[j]["ending_streamlines"].append(i)
                        # new_streamline = torch.cat((cutted_streamline, termination_node.unsqueeze(0)), 0)
                        # print(f"type of new streamlines {type(new_streamline)}{new_streamline.shape}")
                        # Streamlines[i]["coords"] = np.array(new_streamline)
                        Streamlines[i]["coords"] = np.array(cutted_streamline)
                        Streamlines[i]["ending_singularity"] = j
                        Streamlines[i]["ends_at_boundary"] = Singularity[j]["is_boundary"]
                        print('Streamline cutted in function pre_processing')

        return Singularity, Streamlines

    def merge_streamlines(self, Singularity, Streamlines):

        print("\n function merge_streamlines \n")
        merged_pairs = set()

        for key_i in Streamlines.keys():
            if key_i in merged_pairs:
                continue

            if not Streamlines[key_i]["starts_at_boundary"] and not Streamlines[key_i]["ends_at_boundary"]:
                singularity_start_i = Streamlines[key_i]["starting_singularity"]
                singularity_end_i = Streamlines[key_i]["ending_singularity"]

                if singularity_start_i is None or singularity_end_i is None:
                    continue

                for key_j in Streamlines.keys():

                    if not Streamlines[key_j]["starts_at_boundary"] and not Streamlines[key_j]["ends_at_boundary"]:
                        if key_i == key_j or key_j in merged_pairs:
                            continue

                        singularity_start_j = Streamlines[key_j]["starting_singularity"]
                        singularity_end_j = Streamlines[key_j]["ending_singularity"]

                        if singularity_start_j is None or singularity_end_j is None:
                            continue

                        # Check if streamlines can be merged (end-to-start or start-to-end)
                        if (singularity_end_i == singularity_start_j and
                                singularity_start_i == singularity_end_j):

                            streamline_ij = Streamlines[key_i]["coords"]
                            streamline_ji = Streamlines[key_j]["coords"]

                            if streamline_ij is None or streamline_ji is None:
                                continue

                            # Determine correct order and orientation
                            if singularity_end_i == singularity_start_j:
                                # i->j: keep order, may need to flip j
                                merged = self.interpolate_streamlines(streamline_ij, streamline_ji)
                                new_start = singularity_start_i
                                new_end = singularity_end_j
                            else:
                                # j->i: flip order, may need to flip i
                                merged = self.interpolate_streamlines(streamline_ji, streamline_ij)
                                new_start = singularity_start_j
                                new_end = singularity_end_i

                            # Update the first streamline with merged result
                            Streamlines[key_i]["coords"] = merged
                            Streamlines[key_i]["starting_singularity"] = new_start
                            Streamlines[key_i]["ending_singularity"] = new_end

                            # Mark second streamline for removal
                            merged_pairs.add(key_j)

                            # Update singularity references
                            if new_start is not None:
                                if key_j in Singularity[new_start]["starting_streamlines"]:
                                    Singularity[new_start]["starting_streamlines"].remove(key_j)
                                if key_i not in Singularity[new_start]["starting_streamlines"]:
                                    Singularity[new_start]["starting_streamlines"].append(key_i)

                            if new_end is not None:
                                if key_j in Singularity[new_end]["ending_streamlines"]:
                                    Singularity[new_end]["ending_streamlines"].remove(key_j)
                                if key_i not in Singularity[new_end]["ending_streamlines"]:
                                    Singularity[new_end]["ending_streamlines"].append(key_i)

                                break

        # Remove merged streamlines
        for key in merged_pairs:
            del Streamlines[key]
        print(len(merged_pairs))

        merged_streamlines = []
        for key in Streamlines.keys():
            if Streamlines[key]["coords"] is not None:
                merged_streamlines.append(np.array(Streamlines[key]["coords"]))
            else:
                print('streamline has no coodrinates')

        return merged_streamlines

    def interpolate_streamlines(self, streamline_ij, streamline_ji, num_points=100):
        # Convert streamlines to splines
        splines_ij = self.get_streamlines_as_splines([streamline_ij])
        splines_ji = self.get_streamlines_as_splines([streamline_ji])

        # Extract splines (assuming get_streamlines_as_splines returns [tck, u] for each streamline)
        tck_ij, u_ij = splines_ij[0]
        tck_ji, u_ji = splines_ji[0]

        # Generate uniform parameter values
        u_new = np.linspace(0, 1, num_points)

        # Evaluate splines at new parameter values
        x_ij, y_ij = splev(u_new, tck_ij)
        x_ji, y_ji = splev(u_new, tck_ji)

        x_merged = (1 - u_new) * x_ij + u_new * x_ji
        y_merged = (1 - u_new) * y_ij + u_new * y_ji
        # Stack and return merged streamline
        merged_streamline = np.vstack((x_merged, y_merged)).T
        return merged_streamline

    def add_graph_attr(self):

        self.quad_mesh.streamline_intersections = self.mesh.streamline_intersections
        self.quad_mesh.edge_subdomain_index = self.edges_subdomain
        self.quad_mesh.edge_subdomain_points = self.edge_points

        self.quad_mesh.triangle_nodes = self.mesh.x
        self.quad_mesh.triangle_faces = self.mesh.faces

    def get_mesh(self):

        nodes           = torch.tensor(self.nodes_subdomain)

        temporary_faces = self.get_faces()
        faces            = self.delete_invalid_faces(temporary_faces, nodes)

        edge_index = torch.cat([faces[:2], faces[1:3], faces[2:4], faces[::2], faces[1::2], faces[::3],], dim=1)
        mesh       = Data(x=nodes, edge_index=edge_index, faces=faces)

        return mesh

    # find_containing_face is a duplicate function of streamline generator
    def find_containing_face(self, point):

        nodes           = self.mesh.x[:, 0:2]
        node_to_face_ID = self.mesh.nodes_faces_ids
        faces           = self.mesh.faces

        distances        = np.linalg.norm(nodes - point, axis=1)
        closest_node_idx = np.argmin(distances)
        possible_faces   = node_to_face_ID[closest_node_idx]

        for face_idx in possible_faces:
            face_vertices = faces[:, face_idx]
            triangle_vertices = nodes[face_vertices, 0:2]
            if self.is_point_in_triangle(point, triangle_vertices):
                return face_idx

        return None

    # is_point_in_triangle is a duplicate function of streamline generator
    def is_point_in_triangle(self, point, vertices):
        v0 = vertices[2] - vertices[0]
        v1 = vertices[1] - vertices[0]
        v2 = point - vertices[0]

        # Compute dot products
        dot00 = torch.dot(v0, v0)
        dot01 = torch.dot(v0, v1)
        dot02 = torch.dot(v0, v2)
        dot11 = torch.dot(v1, v1)
        dot12 = torch.dot(v1, v2)

        # Compute barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return False  # Degenerate triangle

        invDenom = 1 / denom
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        # Check if point is inside the triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def delete_invalid_faces(self, faces_to_check, nodes):
        faces = faces_to_check
        num_faces = faces.size(1)

        invalid_faces = torch.zeros(num_faces)

        # Get the dtype of your nodes to ensure consistency
        nodes_dtype = nodes.dtype

        for i in range(num_faces):
            # Get the vertices of the current quad face
            face_vertices = nodes[faces[:, i]]
            # Calculate the interior angles of the quad
            angles = []

            # For each vertex, calculate the angle
            for j in range(4):
                # Get the previous, current, and next vertices (with wrapping)
                prev = face_vertices[(j - 1) % 4]
                curr = face_vertices[j]
                next_v = face_vertices[(j + 1) % 4]

                # Calculate vectors with explicit dtype
                v1 = torch.tensor([prev[0] - curr[0], prev[1] - curr[1]], dtype=nodes_dtype)
                v2 = torch.tensor([next_v[0] - curr[0], next_v[1] - curr[1]], dtype=nodes_dtype)

                # Calculate the angle using dot product and magnitude
                dot_product = torch.dot(v1, v2)
                magnitude = torch.norm(v1) * torch.norm(v2)

                # Avoid division by zero
                if magnitude < 1e-10:
                    angle = torch.tensor(0)
                else:
                    # Get the angle in radians and convert to degrees
                    angle = torch.acos(torch.clamp(dot_product / magnitude, -1.0, 1.0))
                    angle = angle * 180 / torch.pi

                angles.append(angle.item())

            # Check if any angle is close to or greater than 180 degrees
            max_angle = max(angles)
            if max_angle > 175:  # Threshold for "close to 180 degrees"
                invalid_faces[i] = 1

        mask_valid_faces = invalid_faces == 0
        valid_faces = faces[:, mask_valid_faces]

        return valid_faces

    def get_faces(self):
        num_edges = self.edges_subdomain[0].size
        edges = []
        for i in range(num_edges):
            edge = (self.edges_subdomain[0][i], self.edges_subdomain[1][i])
            edges.append(edge)
        graph = nx.Graph()

        graph.add_edges_from(edges)

        faces = list(nx.simple_cycles(graph, length_bound=4))

        quad_faces = []

        for i in range(len(faces)):
            if len(faces[i]) == 4:
                quad_faces.append(faces[i])

        if len(quad_faces) == 0:
            print('non quad face found')

        return torch.tensor(quad_faces).T

    def get_streamlines_as_splines(self, streamlines=None):

        splines = []

        if streamlines == None:
            streamlines = self.mesh.streamlines

        for i in range(len(streamlines)):
            streamline = np.array(streamlines[i])
            x = streamline[:, 0]
            y = streamline[:, 1]
            if x.size == 2:
                tck, u = splprep([x, y], s=0, k=1)  # k=1 linear splines
                splines.append([tck, u])
            else:
                tck, u = splprep([x, y], s=0)  # Cubic Splines (Default)
                splines.append([tck, u])

        return splines

    def reconstruct_streamlines_from_edges(self, split_data):

        streamlines = []
        edges = split_data['edges']

        for edge in edges:
            edge = np.array(edge)
            x, y = edge[:, 0], edge[:, 1]

            # Handle short or straight segments with k=1 (linear)
            if len(edge) <= 2:
                k = 1
            else:
                k = min(3, len(edge) - 1)

            try:
                tck, u = splprep([x, y], s=0, k=k)
                t_vals = np.linspace(0, 1, 100)
                spline_points = np.array(splev(t_vals, tck)).T  # shape (100, 2)
            except Exception as e:
                # fallback: just interpolate linearly
                spline_points = np.linspace(edge[0], edge[-1], 100)

            streamlines.append(spline_points)

        return streamlines

    def split_splines_at_intersections(self, intersection_data):
        """
        Returns:
        Dictionary with:
            - 'edges': List of edges where each edge is a list of points
            - 'edge_to_parent': Dict mapping edge_idx to original spline_idx
            - 'nodes': List of intersection nodes
        """

        splines = self.streamline_splines

        # Extract the dictionary from the tuple
        intersection_dict       = intersection_data
        points                  = intersection_dict['points']
        spline_intersections    = intersection_dict['spline_intersections']

        edges = []
        edge_to_parent = {}

        # Process each spline
        for spline_idx, spline in enumerate(splines):
            if spline is None:
                continue

            tck, u = spline

            # Get all intersection parameters for this spline, sorted by t
            intersections = []
            if spline_idx in spline_intersections:
                # Each intersection is (point_idx, t)
                intersections = sorted(spline_intersections[spline_idx], key=lambda x: x[1])

            # If no intersections, add the entire spline as one edge
            if not intersections:
                t_values = np.linspace(0, 1, 50)  # Sample points along the spline
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx
                continue

            # Add endpoints as special cases (t=0 and t=1)
            # Check if the first intersection is not at t=0
            if intersections[0][1] > 0.001:  # Small tolerance
                t_values = np.linspace(0, intersections[0][1], 20)
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx

            # Create edges between consecutive intersections
            for i in range(len(intersections) - 1):
                point_idx1, t1 = intersections[i]
                point_idx2, t2 = intersections[i + 1]

                # Skip if they're too close
                if abs(t2 - t1) < 0.001:
                    continue

                t_values = np.linspace(t1, t2, max(2, int((t2 - t1) * 50)))
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx

            # Check if the last intersection is not at t=1
            if intersections[-1][1] < 0.999:  # Small tolerance
                t_values = np.linspace(intersections[-1][1], 1, 20)
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx

        # Create a convenient list of nodes for reference
        nodes = [tuple(point) for point in points]

        return {
            'edges': edges,
            'edge_to_parent': edge_to_parent,
            'nodes': nodes
        }

    def get_intersections(self):

        intersections = []
        for i in range(len(self.streamline_splines)):
            streamline_I = self.streamline_splines[i]
            for j in range(i + 1, len(self.streamline_splines)):

                streamline_J = self.streamline_splines[j]

                confirmed_intersection = self.find_spline_intersections_with_params(streamline_I, streamline_J)

                if confirmed_intersection:

                    intersections.append(confirmed_intersection[0])

        return intersections

    def find_all_intersections(self, tolerance=1e-6):

        splines = self.streamline_splines
        all_intersections = []
        intersections = []
        for i in range(len(splines)):
            for j in range(i + 1, len(splines)):
                if splines[i] is None or splines[j] is None:
                    continue

                intersections = self.find_spline_intersections_with_params(splines[i], splines[j], tolerance)

                for point, t1, t2 in intersections:
                    all_intersections.append((point, i, t1, j, t2))

        unique_points = []
        point_indices = {}

        spline_intersections = defaultdict(list)  # Maps spline_idx to [(point_idx, t), ...]
        connectivity = defaultdict(list)  # Maps point_idx to [spline_idx, ...]

        for point, spline1_idx, t1, spline2_idx, t2 in all_intersections:
            # Check if this point is already in unique_points
            is_new_point = True
            point_idx = None

            for idx, existing_point in enumerate(unique_points):
                if np.linalg.norm(np.array(point) - np.array(existing_point)) < tolerance:
                    is_new_point = False
                    point_idx = idx
                    break

            if is_new_point:
                point_idx = len(unique_points)
                unique_points.append(point)

            # Update spline_intersections
            spline_intersections[spline1_idx].append((point_idx, t1))
            spline_intersections[spline2_idx].append((point_idx, t2))

            # Update connectivity
            if spline1_idx not in connectivity[point_idx]:
                connectivity[point_idx].append(spline1_idx)
            if spline2_idx not in connectivity[point_idx]:
                connectivity[point_idx].append(spline2_idx)

        # Sort spline_intersections by parameter valuet
        for spline_idx in spline_intersections:
            spline_intersections[spline_idx].sort(key=lambda x: x[1])

        intersection_coords = [torch.tensor(p, dtype=torch.float32) for p in unique_points]

        return {
            'points': unique_points,
            'spline_intersections': dict(spline_intersections),
            'connectivity': dict(connectivity)
        }, intersection_coords

    def find_spline_intersections_with_params(self, spline1, spline2, tolerance=1e-5, num_samples=3):

        offset_boundingBox = 0.05
        tck1, u1 = spline1
        tck2, u2 = spline2

        u1_fine = np.linspace(0, 1, num_samples)
        u2_fine = np.linspace(0, 1, num_samples)

        points1 = np.array(splev(u1_fine, tck1)).T
        points2 = np.array(splev(u2_fine, tck2)).T

        # Find potential intersection regions
        potential_intersections = []

        # For each segment in spline1, check for potential intersections with segments in spline2
        for i in range(len(points1) - 1):
            for j in range(len(points2) - 1):
                # Check if the bounding boxes of the segments overlap
                min_x1, max_x1 = min(points1[i][0], points1[i + 1][0]), max(points1[i][0], points1[i + 1][0])
                min_y1, max_y1 = min(points1[i][1], points1[i + 1][1]), max(points1[i][1], points1[i + 1][1])

                min_x2, max_x2 = min(points2[j][0], points2[j + 1][0]), max(points2[j][0], points2[j + 1][0])
                min_y2, max_y2 = min(points2[j][1], points2[j + 1][1]), max(points2[j][1], points2[j + 1][1])

                # extent Bounding Box to get intersections where start/endpoint are the same
                # example horizonal and vertical line that are starting at the same point will not be detected.

                min_x11 = min_x1 - max(offset_boundingBox, offset_boundingBox * min_x1)
                max_x11 = max_x1 + max(offset_boundingBox, offset_boundingBox * max_x1)

                min_y11 = min_y1 - max(offset_boundingBox, offset_boundingBox * min_y1)
                max_y11 = max_y1 + max(offset_boundingBox, offset_boundingBox * max_y1)

                min_x22 = min_x2 - max(offset_boundingBox, offset_boundingBox * min_x2)
                max_x22 = max_x2 + max(offset_boundingBox, offset_boundingBox * max_x2)

                min_y22 = min_y2 - max(offset_boundingBox, offset_boundingBox * min_y2)
                max_y22 = max_y2 + max(offset_boundingBox, offset_boundingBox * max_y2)

                # If bounding boxes overlap, add to potential intersections
                if (min_x11 <= max_x22 and max_x11 >= min_x22 and
                        min_y11 <= max_y22 and max_y11 >= min_y22):
                    u1_val = u1_fine[i]
                    u2_val = u2_fine[j]
                    potential_intersections.append((u1_val, u2_val))

        confirmed_intersections = []

        for u1_val, u2_val in potential_intersections:
            # Define a function to find the root of (distance between points on the splines)
            def distance_func(params):
                t1, t2 = params
                point1 = np.array(splev(t1, tck1)).reshape(2)
                point2 = np.array(splev(t2, tck2)).reshape(2)

                return [point1[0] - point2[0], point1[1] - point2[1]]

            # Initial guess
            initial_guess = [u1_val, u2_val]

            # Solve for intersection
            try:
                from scipy.optimize import fsolve
                t1_intersect, t2_intersect = fsolve(distance_func, initial_guess)

                # Check if t1_intersect and t2_intersect are within [0, 1]
                if 0 <= t1_intersect <= 1 and 0 <= t2_intersect <= 1:

                    point1 = np.array(splev(t1_intersect, tck1)).reshape(2)
                    point2 = np.array(splev(t2_intersect, tck2)).reshape(2)

                    # Check if points are close enough
                    if np.linalg.norm(point1 - point2) < tolerance:
                        # Average the two points to get the intersection point
                        intersection_point = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

                        # Check if this intersection is already in the list
                        is_duplicate = False
                        for existing_point, _, _ in confirmed_intersections:
                            if np.linalg.norm(np.array(existing_point) - np.array(intersection_point)) < tolerance:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            confirmed_intersections.append((intersection_point, t1_intersect, t2_intersect))
            except:
                # If fsolve fails, skip this potential intersection
                continue

        return confirmed_intersections

    def extract_subdomain_arrays(self, intersection_data):
        """
        Extracts intersection data into NumPy arrays for subdomain creation.
        Improved version that handles spline start/end points as intersections.

        Parameters:
        intersection_data: Tuple containing intersection information

        Returns:
        Dictionary with:
            - 'edges_subdomain': [2 x n_edges] array with indices of connected intersection points
            - 'nodes_subdomain': [n_intersection_nodes x 2] array with x,y coordinates of intersection points
            - 'edge_points': List where each entry is an array of points along the corresponding edge
        """
        # Extract intersection dictionary from tuple
        intersection_dict = intersection_data
        points = intersection_dict['points']
        spline_intersections = intersection_dict['spline_intersections']

        # Create nodes_subdomain: array of node coordinates
        nodes_subdomain = np.array(points)

        # Prepare to build edges_subdomain and edge_points
        edges_list = []
        edge_points = []

        # Get all endpoint coordinates of splines
        spline_endpoints = {}
        for spline_idx, spline in enumerate(self.streamline_splines):
            if spline is None:
                continue
            tck, u = spline
            # Get start and end points
            start_point = tuple(np.array(splev(0, tck)).reshape(2))
            end_point = tuple(np.array(splev(1, tck)).reshape(2))
            spline_endpoints[spline_idx] = (start_point, end_point)

        # Function to find the closest point index in our points list
        def find_closest_point_idx(point, tolerance=1e-6):
            for idx, p in enumerate(points):
                if np.linalg.norm(np.array(point) - np.array(p)) < tolerance:
                    return idx
            return None

        # Process each spline
        for spline_idx in spline_intersections:
            # Get the spline data
            tck, u = self.streamline_splines[spline_idx]

            # Get all intersection points for this spline, sorted by parameter t
            intersections = sorted(spline_intersections[spline_idx], key=lambda x: x[1])

            # Check if start point is an intersection in any spline
            start_point = spline_endpoints[spline_idx][0]
            end_point = spline_endpoints[spline_idx][1]

            # Check if start point is close to first intersection
            start_in_intersections = False
            if intersections:
                first_t = intersections[0][1]
                start_in_intersections = abs(first_t) < 0.001

            # Check if end point is close to last intersection
            end_in_intersections = False
            if intersections:
                last_t = intersections[-1][1]
                end_in_intersections = abs(last_t - 1) < 0.001

            # Find point indices for start and end points
            start_idx = find_closest_point_idx(start_point)
            end_idx = find_closest_point_idx(end_point)

            # Create edges between consecutive intersection points on the same spline
            if intersections:
                for i in range(len(intersections) - 1):
                    point_idx1, t1 = intersections[i]
                    point_idx2, t2 = intersections[i + 1]

                    # # Skip if they're too close
                    # if abs(t2 - t1) < 0.001:
                    #     continue

                    # Add the edge indices
                    edges_list.append([point_idx1, point_idx2])

                    # Generate points along this edge segment
                    num_points = max(10, int((t2 - t1) * 50))  # Adjust number of points based on parameter length
                    t_values = np.linspace(t1, t2, num_points)
                    edge_segment_points = np.array(splev(t_values, tck)).T
                    edge_points.append(edge_segment_points)

                # Handle start point to first intersection if needed
                if start_idx is not None and not start_in_intersections and intersections[0][1] > 0.001:
                    point_idx1 = start_idx
                    point_idx2, t2 = intersections[0]
                    edges_list.append([point_idx1, point_idx2])
                    t_values = np.linspace(0, t2, max(10, int(t2 * 50)))
                    edge_segment_points = np.array(splev(t_values, tck)).T
                    edge_points.append(edge_segment_points)

                # Handle last intersection to end point if needed
                if end_idx is not None and not end_in_intersections and intersections[-1][1] < 0.999:
                    point_idx1, t1 = intersections[-1]
                    point_idx2 = end_idx
                    edges_list.append([point_idx1, point_idx2])
                    t_values = np.linspace(t1, 1, max(10, int((1 - t1) * 50)))
                    edge_segment_points = np.array(splev(t_values, tck)).T
                    edge_points.append(edge_segment_points)

            # If no intersections on this spline, but start/end points are intersections elsewhere,
            # create an edge between them
            elif start_idx is not None and end_idx is not None:
                edges_list.append([start_idx, end_idx])
                t_values = np.linspace(0, 1, 50)
                edge_segment_points = np.array(splev(t_values, tck)).T
                edge_points.append(edge_segment_points)

        # Check for any missing edges between intersection points
        # Find splines that might have been missed
        for spline_idx, spline in enumerate(self.streamline_splines):
            if spline is None or spline_idx in spline_intersections:
                continue

            # Get endpoints
            start_point, end_point = spline_endpoints[spline_idx]
            start_idx = find_closest_point_idx(start_point)
            end_idx = find_closest_point_idx(end_point)

            # If both endpoints are intersections, add an edge between them
            if start_idx is not None and end_idx is not None:
                edges_list.append([start_idx, end_idx])
                tck, u = spline
                t_values = np.linspace(0, 1, 50)
                edge_segment_points = np.array(splev(t_values, tck)).T
                edge_points.append(edge_segment_points)

        # Convert to NumPy array and transpose to get [2 x n_edges]
        if edges_list:
            edges_subdomain = np.array(edges_list).T
        else:
            edges_subdomain = np.zeros((2, 0), dtype=int)

        return edges_subdomain, nodes_subdomain, edge_points
