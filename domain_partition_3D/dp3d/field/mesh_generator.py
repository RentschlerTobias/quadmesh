from os import write
import gmsh
import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_isolated_nodes
from torch_geometric.data import Data


class MeshGenerator:
    def __init__(self, airfoil, lc=0.5, quadMesh=False):
        self.is_quad_mesh = quadMesh
        self.mesh = self.get_mesh_of_airfoil(lc, airfoil)
        self.mesh.nodes_faces_ids = self.map_nodes_to_faces()
        self.mesh.edge_attr = self.add_edge_attr()
        self.mesh.face_attr = self.add_face_attr()

        self.mesh = self.getFaceCenterPoints()
        self.normalize_mesh_coordinates()
    
    def export_to_obj(self, filename="mesh.obj"):
            nodes = self.mesh.x
            nodes[:, 2] = 0
            faces = self.mesh.faces
    
            nodes = nodes.numpy()
            faces = faces.numpy().T
    
            with open(filename, 'w') as file:
                for node in nodes:
                    file.write(f"v {node[0]} {node[1]} {node[2]}\n")
    
                if faces.shape[1] == 3:  # Triangular faces
                    for face in faces:
                        file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                elif faces.shape[1] == 4:  # Quadrilateral faces
                    file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1} {face[3] + 1}\n")
    
            print(f"Mesh exported to {filename}")

    def get_mesh_of_airfoil(self, lc, airfoil):
        gmsh.initialize()
        gmsh.model.add("Airfoil Mesh")
        occ = gmsh.model.occ
        boundary_points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]
        boundary_tags = []
        for i, point in enumerate(boundary_points):
            boundary_tags.append(occ.addPoint(*point, lc, tag=i + 1))
        boundary_lines = [
            occ.addLine(boundary_tags[0], boundary_tags[1]),
            occ.addLine(boundary_tags[1], boundary_tags[2]),
            occ.addLine(boundary_tags[2], boundary_tags[3]),
            occ.addLine(boundary_tags[3], boundary_tags[0])
        ]
        outer_loop = occ.addCurveLoop(boundary_lines)
        streamlines =[]
        num_points = len(boundary_points)
        for i in range(num_points):
            start_point = boundary_points[i][:2]  # (x, y) of the first point
            end_point   = boundary_points[(i + 1) % num_points][:2]  # (x, y) of the next point
            boundary_streamline = []
            boundary_streamline.append(np.array(start_point))  # First point
            boundary_streamline.append(np.array(end_point))    # Second point
            
            streamlines.append(np.array(boundary_streamline))
        # Add airfoil geometry as a spline
        suction_points = []
        pressure_points = []
        for i, point in enumerate(airfoil.suction_side_rotated):
            suction_points.append(occ.addPoint(point[0], point[1], 0, lc))
        for i, point in enumerate(airfoil.pressure_side_rotated):
            pressure_points.append(occ.addPoint(point[0], point[1], 0, lc))

        suction_spline = occ.addSpline(suction_points)
        pressure_spline = occ.addSpline(pressure_points)
        airfoil_loop = occ.addCurveLoop([suction_spline, pressure_spline])

        streamlines.append(np.array(airfoil.suction_side_rotated))
        streamlines.append(np.array(airfoil.pressure_side_rotated))
        # Add the plane surface
        plane_surface = occ.addPlaneSurface([outer_loop, airfoil_loop])
        occ.synchronize()
        if self.is_quad_mesh == True:
            # Mesh settings
            gmsh.option.setNumber("Mesh.Algorithm", 5)  # MeshAdapt algorithm
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)  # Smoothing steps

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)
        node_coords_tensor = torch.from_numpy(node_coords).float()
        node_tags = node_tags - 1  # Convert to 0-based indexing

        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()

        faces = None

        # Check for triangles and quads
        for etype, etags, ntags in zip(element_types, element_tags, node_tags_per_element):
            if etype == 2:  # Triangles
                faces = np.array(ntags).reshape(-1, 3) - \
                    1  # Convert to 0-based indexing
                break
            elif etype == 3:  # Quadrilaterals
                faces = np.array(ntags).reshape(-1, 4) - \
                    1  # Convert to 0-based indexing

        if faces is None:
            raise ValueError(
                "No triangular or quadrilateral elements found in the mesh.")

        # Convert faces to PyTorch format
        faces_tensor = torch.tensor(faces.T.astype(
            np.int64), dtype=torch.long)  # Transpose and cast to int64

        # Remove isolated nodes and adjust new node indices in faces_tensor
        edge_index = self.face_to_edges(faces_tensor)
        new_edge_index, _, mask = remove_isolated_nodes(edge_index)

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
       # Process node dimensions if needed
        for i in range(node_coords_tensor.size(0)):
            nodeTag = i + 1
            coord, _, dim, tag = gmsh.model.mesh.getNode(nodeTag)
            node_coords_tensor[i, 2] = dim
 
        # Update node coordinates based on mask
        new_node_coords = node_coords_tensor[mask, :]

        gmsh.finalize()

        mesh = Data(x=new_node_coords, edge_index=new_edge_index,
                    faces=updated_faces, streamlines = streamlines)

        return mesh

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

    def normalize_mesh_coordinates(self):
        nodeDim = self.mesh.x.size()[0]
        transformValues = torch.zeros(nodeDim, dtype=torch.float)

        xMin = (torch.min(self.mesh.x, dim=0)).values[0]
        xMax = (torch.max(self.mesh.x, dim=0)).values[0]
        yMin = (torch.min(self.mesh.x, dim=0)).values[1]
        yMax = (torch.max(self.mesh.x, dim=0)).values[1]

        transformValues[0] = xMin
        transformValues[1] = xMax
        transformValues[2] = yMin
        transformValues[3] = yMax

        normNodes = torch.zeros((nodeDim, 2), dtype=torch.float)
        normNodes[:, 0] = (self.mesh.x[:, 0]-xMin)/(xMax-xMin)
        normNodes[:, 1] = (self.mesh.x[:, 1]-yMin)/(yMax-yMin)

        # num_center_nodes = self.mesh.centerPoints.size()[0]
        # normCenterNodes = torch.zeros((num_center_nodes, 2), dtype=torch.float)
        # normCenterNodes[:, 0] = (self.mesh.centerPoints[:, 0]-xMin)/(xMax-xMin)
        # normCenterNodes[:, 1] = (self.mesh.centerPoints[:, 1]-yMin)/(yMax-yMin)
        # self.mesh.centerPoints = normCenterNodes
        self.mesh.x[:, 0:2] = normNodes

    def map_nodes_to_faces(self):
        faces = self.mesh.faces
        node_to_faces = {}
        num_faces = faces.size(1)

        for face_id in range(num_faces):
            nodes_in_face = faces[:, face_id]
            for node_id in nodes_in_face.tolist():
                if node_id not in node_to_faces:
                    node_to_faces[node_id] = []
                node_to_faces[node_id].append(face_id)

        return node_to_faces

   
    def add_edge_attr(self):
        
        # all edges which are part of only one face (triangle) are boundary edges
        faces = self.mesh.faces
        edge_index = self.mesh.edge_index
        edges_of_faces = torch.cat([
                    faces[0:2, :],  # Edges from vertex 0 to vertex 1
                    faces[1:3, :],  # Edges from vertex 1 to vertex 2
                    faces[[2, 0], :]  # Edges from vertex 2 back to vertex 0
                ], dim=1)
        edges_of_faces = torch.sort(edges_of_faces, dim=0).values

        edges_of_faces = edges_of_faces.T

        unique_edges, counts = torch.unique(edges_of_faces, dim=0, return_counts=True)
        boundary_mask = counts == 1
        boundary_edges = unique_edges[boundary_mask]

        interior_mask = counts == 2
        interior_edges = unique_edges[interior_mask]

        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

        for boundary_edge in boundary_edges:
            idx = torch.where(
                (edge_index[0, :] == boundary_edge[0]) & (edge_index[1, :] == boundary_edge[1]) |
                (edge_index[0, :] == boundary_edge[1]) & (edge_index[1, :] == boundary_edge[0])
            )[0]
            edge_attr[idx] = 1  # Mark as boundary edge
   
        return edge_attr
   
    def add_face_attr(self):

        nodes = self.mesh.x
        num_nodes = nodes.size(0)
        num_faces = self.mesh.faces.size(1)
        faces_attr = torch.zeros(num_faces)
        for i in range(num_nodes):
            node = nodes[i, :]
            if node[2] != 2:
                faces_of_node_i = self.mesh.nodes_faces_ids[i]

                for j in range(len(faces_of_node_i)):
                    face_id = faces_of_node_i[j]
                    faces_attr[face_id] = 1

        #self.mesh.face_attr = faces_attr
        return faces_attr

    def getFaceCenterPoints(self):
        mesh = self.mesh
        faces = mesh.faces
        coordinates = mesh.x[:, 0:2]
        num_faces = faces.shape[1]
        barycenters = torch.zeros((num_faces, 2))

        for i in range(num_faces):
            vertices = faces[:, i]
            vertex_coords = coordinates[vertices]
            barycenter = torch.mean(vertex_coords, dim=0)
            barycenters[i] = barycenter

        mesh.centerPoints = barycenters
        return mesh

    
