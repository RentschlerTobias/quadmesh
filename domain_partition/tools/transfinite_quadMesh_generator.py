from os import write
import gmsh
import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_isolated_nodes
from torch_geometric.data import Data


class QuadMeshGenerator:
    def __init__(self, block_mesh, lc=0.5, transfinite_divisions=20):

        self.nodes = block_mesh.x
        self.faces = block_mesh.faces

        self.streamline_mapping = block_mesh.edge_to_streamline
        self.transfinite_divisions = transfinite_divisions

        self.transfinite_interpolation()
        self.transfinite_mesh = self.gmsh_mesh_to_torch_graph()

    def faces_to_edges(self, faces):
        if faces.size()[0] == 3:
            edges = torch.cat(
                [faces[[0, 1], :], faces[[1, 2], :], faces[[2, 0], :]], dim=1)
        if faces.size()[0] == 4:
            edges = torch.cat([faces[[0, 1], :], faces[[1, 2], :], faces[[
                              2, 3], :], faces[[3, 0], :]], dim=1)

        edges = to_undirected(edges)
        edges = edges.to(torch.long)
        return edges

    def transfinite_interpolation(self):

        gmsh.initialize()
        gmsh.model.add("quad_mesh")
        gmsh.model.mesh.setTransfiniteAutomatic(
            [], cornerAngle=2.35, recombine=True)

        points = {}
        for i, face in enumerate(self.faces.T):
            curves = []
            for j in range(4):
                edge = (face[j].item(), face[(j + 1) % 4].item())
                streamline = self.streamline_mapping[edge]

                curve_points = []
                for pt in streamline:
                    key = (round(pt[0], 3), round(pt[1], 3))
                    if key not in points:
                        points[key] = gmsh.model.geo.addPoint(pt[0], pt[1], 0)
                    curve_points.append(points[key])

                if len(curve_points) == 2:
                    curve_id = gmsh.model.geo.addLine(
                        curve_points[0], curve_points[1])
                else:
                    curve_id = gmsh.model.geo.addSpline(curve_points)

                curves.append(curve_id)
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    curve_id, self.transfinite_divisions)

            loop = gmsh.model.geo.addCurveLoop(curves)
            surf = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.mesh.setTransfiniteSurface(surf, "Alternate")
            gmsh.model.geo.mesh.setRecombine(2, surf)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

    def gmsh_mesh_to_torch_graph(self):

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)
        node_coords_tensor = torch.from_numpy(node_coords).float()
        node_tags = node_tags - 1  # Convert to 0-based indexing

        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()

        faces = None

        # Check for triangles and quads
        new_faces = None
        for etype, etags, ntags in zip(element_types, element_tags, node_tags_per_element):
            if etype == 3:  # Quadrilaterals - prioritize these
                # Convert to 0-based indexing
                new_faces = np.array(ntags).reshape(-1, 4) - 1
                break
            elif etype == 2 and faces is None:  # Only use triangles if no quads are found
                # Convert to 0-based indexing
                new_faces = np.array(ntags).reshape(-1, 3) - 1

        if new_faces is None:
            raise ValueError(
                "No triangular or quadrilateral elements found in the mesh.")

        # Convert faces to PyTorch format
        faces_tensor = torch.tensor(new_faces.T.astype(
            np.int64), dtype=torch.long)  # Transpose and cast to int64

        # Remove isolated nodes and adjust new node indices in faces_tensor
        edge_index = self.faces_to_edges(faces_tensor)
        num_nodes = node_coords_tensor.size(0)

        new_edge_index, _, mask = remove_isolated_nodes(
            edge_index, num_nodes=num_nodes)

        # Create a mapping for old to new indices
        index_mapping = torch.full(
            (node_coords_tensor.size(0),), -1, dtype=torch.long)
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

        return Data(x=new_node_coords, edge_index=new_edge_index,
                    faces=updated_faces)
