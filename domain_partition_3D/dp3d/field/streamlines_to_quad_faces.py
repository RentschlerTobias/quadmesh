import numpy as np
import torch
from typing import List
from torch_geometric.data import Data
import networkx as nx



class QuadFaceGenerator:

    def __init__(self, streamlines: List, verbose: bool = True):
    
   
        self.streamlines = streamlines


    def get_data(self):
        
        nodes, edges, edges_to_streamline,updated_streamlines  = self.build_connectivity(self.streamlines)

        faces                       = self.get_quad_faces(edges)

        # nodes of faces are not sorted => sort them counter clockwise
        sorted_faces                = self.sort_faces_ccw(faces,nodes)
        
        # map the streamlines to the corresponding edges

        edge_to_streamline         = self.map_face_edges_streamline(sorted_faces,nodes,edges_to_streamline)
        
        edge_index                 = self.faces_to_edges(sorted_faces)
        faces                      = sorted_faces

        return faces, edge_to_streamline,edge_index, nodes


    def build_connectivity(self,streamlines, tol=1e-2):
        # Collect all unique nodes
        nodes = {}
        for s in streamlines:
            for p in [tuple(s[0]), tuple(s[-1])]:
                if not any(np.allclose(p, n, atol=tol) for n in nodes):
                    nodes[p] = len(nodes)
        
        # Build edges and flip duplicates
        node_list = list(nodes.keys())
        edges = []  # [start_node, end_node] pairs
        used_pairs = set()
        processed_streamlines = []  # Store potentially flipped streamlines
        edges_to_streamline = {}
        for i, s in enumerate(streamlines):
            start_idx = next(j for j, n in enumerate(node_list) if np.allclose(s[0], n, atol=tol))
            end_idx = next(j for j, n in enumerate(node_list) if np.allclose(s[-1], n, atol=tol))
         
            # Check if this node pair already exists
            if (start_idx, end_idx) in used_pairs:
                # Flip the streamline
                streamline_reversed = s[::-1]
                processed_streamlines.append(streamline_reversed)  # Reverse the streamline
                edges.append([end_idx, start_idx])
                used_pairs.add((end_idx, start_idx))
                edges_to_streamline[(end_idx, start_idx)] = streamline_reversed
            else:
                processed_streamlines.append(s)  # Keep original
                edges.append([start_idx, end_idx])
                used_pairs.add((start_idx, end_idx))
                edges_to_streamline[(start_idx,end_idx)] = s
    
        
        return (torch.tensor(node_list),           # (n_nodes, 2)
                torch.tensor(edges).T,            
                edges_to_streamline,
                processed_streamlines)             

    def get_quad_faces(self,edges):
        Edges = []
        for e in range(edges.size(1)):
            Edges.append((edges[0,e].item(),edges[1,e].item()))

        graph = nx.Graph()
        graph.add_edges_from(Edges)

        # Extract actual planar faces (regions), not arbitrary 4-cycles.
        # simple_cycles returns every chordless/chorded <=4 cycle in the graph,
        # including spurious diagonal quads -> inconsistent partition. The real
        # block faces are the bounded regions of the planar embedding.
        is_planar, embedding = nx.check_planarity(graph)
        if not is_planar:
            # Graph not planar -> partition is broken, no valid quad faces.
            return torch.empty((4, 0), dtype=torch.long)

        faces_raw = []
        seen_half_edges = set()
        for u, v in embedding.edges():
            if (u, v) in seen_half_edges:
                continue
            face = embedding.traverse_face(u, v, mark_half_edges=seen_half_edges)
            faces_raw.append(face)

        # Keep only quad regions. The unbounded outer face (rectangle + airfoil
        # contour) has >4 nodes and is naturally dropped by the length filter.
        quad_faces = [face for face in faces_raw if len(face) == 4]

        if len(quad_faces) == 0:
            return torch.empty((4, 0), dtype=torch.long)

        return torch.tensor(quad_faces).T

    def sort_faces_ccw(self,faces, nodes):
        sorted_faces = faces.clone()
        
        for i in range(faces.size(1)):
            face = faces[:, i]
            coords = nodes[face, :].numpy()
            
            # Shoelace formula for signed area
            signed_area = 0
            for j in range(4):
                k = (j + 1) % 4
                signed_area += coords[j, 0] * coords[k, 1] - coords[k, 0] * coords[j, 1]
            
            # If clockwise (negative area), flip the face
            if signed_area < 0:
                sorted_faces[:, i] = torch.flip(face, [0])
        
        return sorted_faces

    def faces_to_edges(self,faces):
        
        if faces.size()[0] == 3:
                edge_index = torch.cat([
                    faces[:2],
                    faces[1:],
                    faces[::2],
                ], dim=1)
        else:
                assert faces.size()[0] == 4
                edge_index = torch.cat([
                    faces[:2],
                    faces[1:3],
                    faces[2:4],
                    faces[::2],
                    faces[1::2],
                    faces[::3],
                ], dim=1)

        return edge_index
    def map_face_edges_streamline(self,faces, nodes, edge_streamline_mapping):

        face_edge_to_streamline = {}
        
        for i in range(faces.size(1)):
            face = faces[:, i]
            face_nodes = nodes[face, :]
            center = torch.mean(face_nodes, dim=0)
            
            for j in range(4):
                start = face[j].item()
                end = face[(j+1) % 4].item()
                edge = (start, end)
                edge_reversed = (end, start)
                
                candidates = []
                if edge in edge_streamline_mapping:
                    candidates.append(edge_streamline_mapping[edge])
                if edge_reversed in edge_streamline_mapping:
                    candidates.append(edge_streamline_mapping[edge_reversed][::-1].copy())
                
                if len(candidates) == 1:
                    face_edge_to_streamline[edge] = candidates[0]
                elif len(candidates) > 1:
                    best_distance = torch.inf
                    best_streamline = None
                    
                    for candidate in candidates:
                        # candidate_tensor = torch.tensor(candidate, dtype=torch.float32)
                        candidate_tensor = torch.from_numpy(candidate.copy()).float()
                        distances = torch.linalg.norm(candidate_tensor - center.unsqueeze(0), dim=1)
                        min_distance = torch.min(distances)
                        
                        if min_distance < best_distance:
                            best_distance = min_distance
                            best_streamline = candidate
                    
                    face_edge_to_streamline[edge] = best_streamline
        
        return face_edge_to_streamline
