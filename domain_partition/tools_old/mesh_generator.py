import gmsh
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from torch_geometric.utils import remove_isolated_nodes


class MeshGenerator:
    def __init__(self, lc=0.5, minBoundary=-5, maxBoundary=5, seed=None):
        self.lc = lc
        self.minBoundary = minBoundary
        self.maxBoundary = maxBoundary
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate the mesh upon initialization
        self.mesh = self.get_mesh()
        self.normalize_mesh_coordinates()
        self.add_edge_attr()
        self.add_face_attr()

    def get_mesh(self):
        offset = np.random.uniform(-0.5, 0.5, 11)
        gmsh.initialize()
        occ = gmsh.model.occ

        # left spline
        occ.addPoint(+0.0 + offset[0], +0.0 + offset[1], 0.0, self.lc, 1)
        occ.addPoint(-3.0 + offset[2], -1.0 + offset[2], 0.0, self.lc, 2)
        occ.addPoint(-1.0 + offset[3], -2.0 + offset[4], 0.0, self.lc, 3)
        occ.addPoint(+0.0 + offset[5], -2.0 + offset[6], 0.0, self.lc, 4)
        occ.addBSpline([1, 2, 3, 4,], degree=2, tag=101)

        # right spline
        occ.addPoint(+0.0 + offset[0], +0.0 + offset[1], 0.0, self.lc, 5)
        occ.addPoint(+2.0 + offset[7], -1.0 + offset[8], 0.0, self.lc, 6)
        occ.addPoint(+1.0 + offset[9], -2.0 + offset[10], 0.0, self.lc, 7)
        occ.addPoint(+0.0 + offset[5], -2.0 + offset[6], 0.0, self.lc, 8)
        occ.addBSpline([5, 6, 7, 8,], degree=2, tag=102)

        # outter boundary
        occ.addPoint(self.maxBoundary, self.maxBoundary, 0.0, self.lc, 25)
        occ.addPoint(self.maxBoundary, self.minBoundary, 0.0, self.lc, 26)
        occ.addPoint(self.minBoundary, self.minBoundary, 0.0, self.lc, 27)
        occ.addPoint(self.minBoundary, self.maxBoundary, 0.0, self.lc, 28)
        occ.addLine(25, 26, tag=201)
        occ.addLine(26, 27, tag=202)
        occ.addLine(27, 28, tag=203)
        occ.addLine(28, 25, tag=204)

        # outter boundary curve loop
        occ.addCurveLoop([201, 202, 203, 204,], 1001)
        # droplet curve loop
        occ.addCurveLoop([101, 102], 1002)

        occ.addPlaneSurface([1001, 1002], 10000)

        occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.createFaces()

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = node_tags - 1
        node_coords_array = np.array(node_coords).reshape(-1, 3)
        nodes_tensor = torch.from_numpy(node_coords_array).float()

        faceTags, faceNodes = gmsh.model.mesh.getAllFaces(3)
        faceTags = faceTags - 1
        faceNodes = faceNodes - 1
        face_nodes_array = np.array(faceNodes).reshape(-1, 3).astype(np.int64)
        faces_tensor = torch.from_numpy(face_nodes_array.T)
        faces_tensor = faces_tensor.to(torch.long)

        edge_index = self.face_to_edges(faces_tensor)
        new_edge_index, _, mask = remove_isolated_nodes(edge_index)
        new_faces = self.adjust_face_indices(faces_tensor, mask)
        num_nodes = nodes_tensor.size(0)
        for i in range(num_nodes):
            nodeTag = i+1
            coord, _, dim, tag = gmsh.model.mesh.getNode(nodeTag)
            nodes_tensor[i, 2] = dim
        nodes = nodes_tensor[mask, :]
        mesh = Data(x=nodes, edge_index=new_edge_index, faces=new_faces)
        nodes_faces_ids = self.map_nodes_to_faces(mesh.faces)
        mesh.nodes_faces_ids = nodes_faces_ids
        mesh = self.getFaceCenterPoints(mesh)
        gmsh.clear()
        gmsh.finalize()
        return mesh

    def normalize_mesh_coordinates(self):
        nodeDim = self.mesh.x.size()[0]
        transformValues = torch.zeros(nodeDim,dtype=torch.float)

        xMin = (torch.min(self.mesh.x,dim=0)).values[0]
        xMax = (torch.max(self.mesh.x,dim=0)).values[0]
        yMin = (torch.min(self.mesh.x,dim=0)).values[1]
        yMax = (torch.max(self.mesh.x,dim=0)).values[1]

        transformValues[0]= xMin
        transformValues[1]= xMax
        transformValues[2]= yMin
        transformValues[3]= yMax

        normNodes      = torch.zeros((nodeDim,2),dtype=torch.float)
        normNodes[:,0] = (self.mesh.x[:,0]-xMin)/(xMax-xMin)
        normNodes[:,1] = (self.mesh.x[:,1]-yMin)/(yMax-yMin)
        
        num_center_nodes      = self.mesh.centerPoints.size()[0]
        normCenterNodes       = torch.zeros((num_center_nodes,2),dtype=torch.float)
        normCenterNodes[:,0]  = (self.mesh.centerPoints[:,0]-xMin)/(xMax-xMin)
        normCenterNodes[:,1]  = (self.mesh.centerPoints[:,1]-yMin)/(yMax-yMin)
        self.mesh.centerPoints    = normCenterNodes
        self.mesh.x[:,0:2]        = normNodes
    
    def face_to_edges(self, faces):
        if faces.size(0) == 3:
            edges = torch.cat(
                [faces[[0, 1], :], faces[[1, 2], :], faces[[2, 0], :]], dim=1)
        elif faces.size(0) == 4:
            edges = torch.cat(
                [faces[[0, 1], :], faces[[1, 2], :], faces[[2, 3], :], faces[[3, 0], :]], dim=1)

        edges = to_undirected(edges)
        return edges.to(torch.long)

    def adjust_face_indices(self, faces, mask):
        index_mapping = torch.full((mask.size(0),), -1, dtype=torch.long)
        index_mapping[mask] = torch.arange(mask.sum())

        valid_faces = ((index_mapping[faces] >= 0).all(dim=0))
        filtered_faces = faces[:, valid_faces]
        updated_faces = index_mapping[filtered_faces]

        return updated_faces

    def getFaceCenterPoints(self, mesh):
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

    def map_nodes_to_faces(self, faces):
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
        edges = self.mesh.edge_index
        numberOfEdges = edges.size()[1]
        edge_attr = torch.zeros([1, numberOfEdges], dtype=torch.long)

        minBoundary = torch.min(self.mesh.x[:, 0])
        maxBoundary = torch.max(self.mesh.x[:, 0])

        for i in range(numberOfEdges):
            node0 = self.mesh.x[edges[0, i], :]
            node1 = self.mesh.x[edges[1, i], :]
            deltaX = node0[0]-node1[0]
            deltaY = node0[1]-node1[1]
            dimNode0 = node0[2]
            dimNode1 = node1[2]

            if dimNode0 == 2 or dimNode1 == 2:
                edge_attr[0, i] = 0
            else:
                if node0[0] == minBoundary or node0[0] == maxBoundary or node0[1] == minBoundary or node0[1] == maxBoundary or node1[0] == minBoundary or node1[0] == maxBoundary or node1[1] == minBoundary or node1[1] == maxBoundary:
                    if deltaX != 0 and deltaY != 0:
                        edge_attr[0, i] = 0
                    else:
                        edge_attr[0, i] = 1

                else:
                    edge_attr[0, i] = 1

        self.mesh.edge_attr = edge_attr

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

        self.mesh.face_attr = faces_attr
