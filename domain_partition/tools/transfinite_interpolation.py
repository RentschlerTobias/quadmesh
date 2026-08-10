import gmsh
import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_isolated_nodes
from torch_geometric.data import Data


class Transfinite_Interpolation:
    def __init__(self, blocked_mesh, mesh_size=0.4):

        self.blocked_mesh = blocked_mesh
        self.nodes = self.blocked_mesh.x

        self.faces = self.blocked_mesh.faces

        self.edge_index = self.blocked_mesh.edge_subdomain_index
        self.edge_points = self.blocked_mesh.edge_subdomain_points

        self.mesh_size = mesh_size

        gmsh.initialize()
        gmsh.model.add("transfinite_quad_mesh")

        self.point_tag_map = {}   # Maps node index to Gmsh point tag
        self.curve_tag_list = []  # Stores Gmsh curve tag for each edge

        self.generate()
        self.quad_mesh = self.get_mesh()

    def face_to_edges(self, faces):
        if faces.size()[0] == 3:
            edges = torch.cat(
                [faces[[0, 1], :], faces[[1, 2], :], faces[[2, 0], :]], dim=1)
        if faces.size()[0] == 4:
            edges = torch.cat([faces[[0, 1], :], faces[[1, 2], :], faces[[
                              2, 3], :], faces[[3, 0], :]], dim=1)

        edges = to_undirected(edges)
        edges = edges.to(torch.long)
        return edges

    def _find_edge_index(self, u, v):
        # Ensure edge_index is a tensor
        if isinstance(self.edge_index, np.ndarray):
            edge_index_tensor = torch.from_numpy(self.edge_index)
        else:
            edge_index_tensor = self.edge_index

        # Search edge_index for both (u, v) and (v, u)
        cond1 = (edge_index_tensor[0] == u) & (edge_index_tensor[1] == v)
        cond2 = (edge_index_tensor[0] == v) & (edge_index_tensor[1] == u)
        result = torch.nonzero(cond1 | cond2)
        if len(result) > 0:
            return result[0].item()
        else:
            return None

    def generate(self, transfinite_divisions=10):
        
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Enable quad recombination
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Use Delaunay triangulation
        gmsh.model.mesh.setTransfiniteAutomatic([], cornerAngle=2.35, recombine=True)
        

        gmsh.model.geo.synchronize()
        point_id_counter = 1
        curve_id_counter = 1

        for i, pt in enumerate(self.nodes):
            tag = gmsh.model.geo.addPoint(pt[0], pt[1], 0, self.mesh_size, point_id_counter)
            self.point_tag_map[i] = tag
            point_id_counter += 1

        for i in range(self.edge_index.shape[1]):
            start_idx = self.edge_index[0, i].item()
            end_idx = self.edge_index[1, i].item()
            start_tag = self.point_tag_map[start_idx]
            end_tag = self.point_tag_map[end_idx]

            edge_points = self.edge_points[i]

            # Only add intermediate points if there are more than just start and end
            if len(edge_points) > 2:
                pt_tags = [start_tag]
                # Add intermediate points (skip first and last which are already added)
                for j in range(1, len(edge_points) - 1):
                    pt = edge_points[j]
                    tag = gmsh.model.geo.addPoint(pt[0], pt[1], 0, self.mesh_size, point_id_counter)
                    pt_tags.append(tag)
                    point_id_counter += 1
                pt_tags.append(end_tag)

                curve_id = gmsh.model.geo.addSpline(pt_tags, curve_id_counter)
            else:
                # Just create a line between start and end points
                curve_id = gmsh.model.geo.addLine(start_tag, end_tag, curve_id_counter)

            self.curve_tag_list.append(curve_id)
            curve_id_counter += 1

        for face_idx, face in enumerate(self.faces.T):
            curve_ids = []
            for i in range(4):
                u = face[i].item()
                v = face[(i + 1) % 4].item()

                edge_idx = self._find_edge_index(u, v)
                if edge_idx is None:
                    raise ValueError(f"Could not find edge between nodes {u} and {v}")

                curve_id = self.curve_tag_list[edge_idx]

                # Check if we need to reverse the curve orientation
                # If the edge direction is opposite to the face traversal direction
                if self.edge_index[0, edge_idx].item() == v and self.edge_index[1, edge_idx].item() == u:
                    curve_id = -curve_id  # Negative sign indicates reversed orientation in GMSH

                curve_ids.append(curve_id)

            try:
                loop_id = gmsh.model.geo.addCurveLoop(curve_ids)
                surface_id = gmsh.model.geo.addPlaneSurface([loop_id])

                gmsh.model.geo.mesh.setTransfiniteSurface(surface_id, "Alternate")
                for cid in curve_ids:
                    gmsh.model.geo.mesh.setTransfiniteCurve(abs(cid), transfinite_divisions)
            except Exception as e:
                print(f"Error creating surface for face {face_idx}: {e}")
                print(f"Face vertices: {face}")
                print(f"Curve IDs: {curve_ids}")
                # Continue with the next face
                continue


        gmsh.model.geo.synchronize()

    def get_mesh(self):

        gmsh.model.mesh.generate(2)
        gmsh.write('./transfinite_quad_mesh.msh')

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)
        node_coords_tensor = torch.from_numpy(node_coords).float()
        node_tags = node_tags - 1  # Convert to 0-based indexing

        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()

        faces = None

        # Check for triangles and quads
        faces = None
        for etype, etags, ntags in zip(element_types, element_tags, node_tags_per_element):
            if etype == 3:  # Quadrilaterals - prioritize these
                faces = np.array(ntags).reshape(-1, 4) - 1  # Convert to 0-based indexing
                break
            elif etype == 2 and faces is None:  # Only use triangles if no quads are found
                faces = np.array(ntags).reshape(-1, 3) - 1  # Convert to 0-based indexing

        if faces is None:
            raise ValueError(
                "No triangular or quadrilateral elements found in the mesh.")

        # Convert faces to PyTorch format
        faces_tensor = torch.tensor(faces.T.astype(
            np.int64), dtype=torch.long)  # Transpose and cast to int64

        # Remove isolated nodes and adjust new node indices in faces_tensor
        edge_index = self.face_to_edges(faces_tensor)
        num_nodes = node_coords_tensor.size(0)

        new_edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=num_nodes)

        # Create a mapping for old to new indices
        index_mapping = torch.full((node_coords_tensor.size(0),), -1, dtype=torch.long)
        index_mapping[mask] = torch.arange(mask.sum(), dtype=torch.long)

        # Adjust faces to remove invalid faces and remap indices
        valid_faces_mask = (index_mapping[faces_tensor] >= 0).all(
            dim=0)  # Check if all nodes in a face are valid
        # Keep only valid faces
        filtered_faces = faces_tensor[:, valid_faces_mask]
        # Remap old indices to new indices
        updated_faces = index_mapping[filtered_faces]
        for i in range(node_coords_tensor.size(0)):
            nodeTag = i + 1
            coord, _, dim, tag = gmsh.model.mesh.getNode(nodeTag)
            node_coords_tensor[i, 2] = dim

        # Update node coordinates based on mask
        new_node_coords = node_coords_tensor[mask, :]

        gmsh.finalize()

        mesh = Data(x=new_node_coords, edge_index=new_edge_index,
                    faces=updated_faces)
        return mesh
