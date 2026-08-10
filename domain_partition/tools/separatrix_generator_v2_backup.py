import torch
import numpy as np
from collections import defaultdict

class SeparatrixGenerator_v2:
    def __init__(self, mesh):

        self.mesh = mesh
        self.detect_faces_with_singularities()
        self.mesh.singularities_coords = {}
        self.get_singularities_coords()
        self.found_separatrices = self.get_separatrices_from_singularity()
        self.check_separatrices_of_singularity()
        self.get_separatrices_from_c0_nodes()

    def detect_faces_with_singularities(self):
        mesh = self.mesh 
        num_faces = mesh.faces.size(1)
        
        vec_x = mesh.frame_field[:, 0]
        vec_y = mesh.frame_field[:, 1]
        angles = torch.atan2(vec_y, vec_x)
        
        angles_of_faces = torch.zeros(3, num_faces)
        angles_of_faces[0, :] = angles[mesh.faces[0, :]]
        angles_of_faces[1, :] = angles[mesh.faces[1, :]]
        angles_of_faces[2, :] = angles[mesh.faces[2, :]]
        
        angles_of_faces_sum = torch.zeros(3, num_faces)
        angles_of_faces_sum[0, :] = torch.remainder(angles_of_faces[1, :] - angles_of_faces[0, :] + torch.pi, 2 * torch.pi) - torch.pi
        angles_of_faces_sum[1, :] = torch.remainder(angles_of_faces[2, :] - angles_of_faces[1, :] + torch.pi, 2 * torch.pi) - torch.pi
        angles_of_faces_sum[2, :] = torch.remainder(angles_of_faces[0, :] - angles_of_faces[2, :] + torch.pi, 2 * torch.pi) - torch.pi
        
        # Compute Poincaré-Index for each Triangle
        poincare_idx = torch.round(torch.sum(angles_of_faces_sum, dim=0) / (2 * torch.pi))
        mesh.singularities = poincare_idx
        
        #Check Euler-Characteristik
        poincare_idx_sum = torch.sum(poincare_idx)
        num_nodes = mesh.x.size(0)
        num_edges = mesh.edge_index.size(1) / 2
        num_faces = mesh.faces.size(1)
        euler_characteristic = num_nodes - num_edges + num_faces

        if euler_characteristic != poincare_idx_sum:
            print(f'Warning: NUmber of  Singularities does not match whith Euler-Characteristik!')
            print(f'Euler-Charakteristik: {euler_characteristic}, Sum Poincaré-Indizes: {poincare_idx_sum}')
        else:
            print('Correct number of Singularities (Euler-Characteristik meet)')
            
        # Expected Num of Separatrices
        # For point care +1 ==> 3 Separatrices, for -1: 5 Separatrices
        mesh.expected_separatrices = {}
        for face_idx in range(num_faces):
            if mesh.singularities[face_idx] != 0:
                if mesh.singularities[face_idx] == 1:  # +1 Singularität
                    mesh.expected_separatrices[face_idx] = 3
                elif mesh.singularities[face_idx] == -1:  # -1 Singularität
                    mesh.expected_separatrices[face_idx] = 5
                else:
                    # Bei höheren Ordnungen entsprechend anpassen
                    mesh.expected_separatrices[face_idx] = abs(int(mesh.singularities[face_idx] * 4 - 1))
                    print(f'expected sing {mesh.expected_separatrices[face_idx] }') 
        return mesh

    def get_singularities_coords(self):

        num_faces = self.mesh.faces.size(1)
        face_all_ids = torch.arange(num_faces)
        face_ids = face_all_ids[self.mesh.singularities != 0]

        for i in range(face_ids.size(0)):
            face_id = face_ids[i].item()
            face = self.mesh.faces[:, face_id]
            vectors = self.mesh.u[face]
            nodes = self.mesh.x[face, 0:2]

            v0, v1, v2 = vectors
            p0, p1, p2 = nodes

            # Interpolation: V(P) = alpha * v0 + beta * v1 + gamma * v2 = 0
            A = torch.stack([v0 - v2, v1 - v2], dim=1)
            b = -v2

            try:
                coeffs = torch.linalg.solve(A, b)
                alpha, beta = coeffs
                gamma = 1.0 - alpha - beta
                
                if alpha >= 0 and beta >= 0 and gamma >= 0 and alpha + beta + gamma <= 1.0 + 1e-5:
                    singularity_location = alpha * p0 + beta * p1 + gamma * p2
                else:
                    singularity_location = torch.mean(nodes, dim=0)
                    print(f"Singularity not in Face {face_id}  -> Use Center of Mass instead")
            except RuntimeError:
                singularity_location = torch.mean(nodes, dim=0)
                print(f"Singularität of Face {face_id} could not be calculated ->  Use Center of Mass instead")

            self.mesh.singularities_coords[face_id] = singularity_location.tolist()

    def get_separatrices_from_singularity(self):

        mesh = self.mesh
        separatrices = []

        # Get Triangles containing a Singularity 
        num_faces = mesh.faces.size(1)
        face_all_ids = torch.arange(num_faces)
        face_ids = face_all_ids[mesh.singularities != 0]

        edge_samples = 100  # number on nodes to investigate for separatrix on each edge 
        tolerance = 1e-2    

        separatrices = []
        for i in range(face_ids.size(0)):
            face_id = face_ids[i].item()
            
            singularity_coords = torch.tensor(mesh.singularities_coords[face_id], dtype=torch.float32)
            
            face_vertices = mesh.faces[:, face_id]
            triangle_points = mesh.x[face_vertices, 0:2]

            
            for edge_idx in range(3):
                found_separatrices = []
                # Start & End of Edge 
                start_idx = edge_idx
                end_idx = (edge_idx + 1) % 3
                
                start_point = triangle_points[start_idx]
                end_point = triangle_points[end_idx]
                
                edge_vector = end_point - start_point

                t_values = torch.linspace(0, 1, edge_samples)
                
                for t in t_values:
                    current_point = start_point + t *edge_vector 
                    
                    direction_to_point = current_point - singularity_coords
                    direction_to_point_norm = torch.norm(direction_to_point)
                       
                    direction_to_point = direction_to_point / direction_to_point_norm
                    
                    # Bestimme den Cross-Vektor am aktuellen Punkt
                    best_vector, _ = self.get_best_cross_vector(current_point, direction_to_point, mesh, face_id)
                    
                    if best_vector is None:
                        continue
                        
                    best_vector = best_vector / torch.norm(best_vector)
                    
                    dot_product = torch.dot(direction_to_point, best_vector)
                    dot_product = torch.clamp(dot_product, -1.0, 1.0)
                    angle_diff = torch.acos(dot_product)
                    
                    if angle_diff < tolerance:
                        abs_angle = torch.atan2(direction_to_point[1], direction_to_point[0])
                        if abs_angle < 0:
                            abs_angle += 2 * torch.pi
                            
                        # Prüfe, ob wir bereits eine ähnliche Separatrix gefunden haben
                        is_new = True
                        for existing_separatrix in found_separatrices:
                            existing_angle = existing_separatrix['angle']
                            if torch.abs(existing_angle - abs_angle) < tolerance:

                                previous_diff = existing_separatrix['angle_diff']
                                print(f'previous_diff {previous_diff} new { angle_diff}')
                                if previous_diff > angle_diff:
                                    print(f' replaced previous_diff {previous_diff} new { angle_diff}')
                                    found_separatrices.remove(existing_separatrix)
                                else:    
                                    is_new = False
                                    break

                        if is_new:
                            found_separatrices.append({
                                'coordinates': current_point,
                                'angle': abs_angle,
                                'vector': best_vector,
                                'singularity_coords': singularity_coords,
                                'face_id': face_id,
                                'angle_diff' : abs_angle
                            })
                separatrices = separatrices +found_separatrices
        return separatrices

    def check_separatrices_of_singularity(self):

        found_separatrices=self.found_separatrices  
        count_separatrices_per_singularity = defaultdict(int)

        mesh = self.mesh
        for separatrix in found_separatrices:
            singularity_face_id = separatrix['face_id']
            count_separatrices_per_singularity[singularity_face_id] += 1


        for face_id, expected_count in mesh.expected_separatrices.items():
            found_count = count_separatrices_per_singularity.get(face_id, 0)

            if found_count != expected_count:
                print(f"Mismatch at face {face_id}: expected {expected_count}, found {found_count}")
    
                
                length_of_edges = []
                face_vertices = mesh.faces[:, face_id]
                triangle_points = mesh.x[face_vertices, 0:2]
        
                for edge_idx in range(3):
                    # Start & End of Edge 
                    start_idx = edge_idx
                    end_idx = (edge_idx + 1) % 3
                    
                    start_point = triangle_points[start_idx]
                    end_point = triangle_points[end_idx]
                    
                    edge_vector = end_point - start_point
                    length_of_edges.append(torch.linalg.norm(edge_vector))
                    
                mean_length = torch.stack(length_of_edges).mean()

                singularity_coords = torch.tensor(self.mesh.singularities_coords[face_id], dtype=torch.float32)

                found_angles = []
                for separatrix in found_separatrices:

                    if separatrix['face_id'] ==face_id:
                        found_angles.append(separatrix['angle'])

                found_angles.sort()
                sorted_angles = found_angles 
                sorted_angles.append(sorted_angles[0]+2*torch.pi)
                sorted_angles = torch.tensor(sorted_angles)
                diff_angles   = torch.diff(sorted_angles) 
                idx_new_angle = torch.where(diff_angles == torch.max(diff_angles))[0]
                new_angle     = (sorted_angles[idx_new_angle]+sorted_angles[idx_new_angle+1])/2 
                new_separatrix_vec = torch.tensor([mean_length*torch.sin(new_angle),mean_length*torch.cos(new_angle)])
                new_separatrix_node = singularity_coords+new_separatrix_vec 

                best_vector, _ = self.get_best_cross_vector(new_separatrix_node, new_separatrix_vec, mesh, face_id)

                found_separatrices.append({
                    'coordinates': new_separatrix_node,
                    'angle': new_angle,
                    'vector': best_vector,
                    'singularity_coords': singularity_coords,
                    'face_id': face_id
                })

                print(f"Added additional separatrix at Face {face_id}")

            else:
                print(f"Face {face_id}: OK ({found_count} separatrices)")

        
        self.mesh.separatrices = found_separatrices

    def get_separatrices_from_c0_nodes(self):
        """Erzeugt Separatrizen von C0-Knoten (Randknoten), die keine regulären Boundary-Knoten sind"""
        mesh = self.mesh
        separatrices = mesh.separatrices
        
        # Identifiziere C0-Knoten (Randknoten mit Label 0)
        mask_c0_nodes = mesh.x[:, 2] == 0
        node_ids = torch.arange(mesh.x.size(0))
        c0_node_ids = node_ids[mask_c0_nodes]
        
        # Parameter
        step = 0.001  # Schrittweite vom Randknoten ins Innere
        
        # Für jeden C0-Knoten
        for i, node_id_tensor in enumerate(c0_node_ids):
            node_id = node_id_tensor.item()
            
            # Überspringe reguläre Boundary-Knoten
            if self.is_boundary_node_regular(node_id):
                continue
                
            # Position des Knotens
            source_node = mesh.x[node_id, 0:2]
            
            # Referenzvektor am Knoten
            ref_vec = mesh.u[node_id]
            base_angle = torch.atan2(ref_vec[1], ref_vec[0]) / 4  # /4 für Cross-Field
            
            # Generiere die 4 Cross-Vektoren
            for k in range(4):
                angle = base_angle + k * torch.pi / 2
                vec = torch.tensor([torch.cos(angle), torch.sin(angle)])
                
                # Erzeuge einen Punkt leicht im Inneren des Meshs
                next_node = source_node + step * vec
                next_node = next_node.to(torch.float)
                
                # Prüfe, ob der Punkt im Mesh liegt
                if self.is_point_inside_mesh(next_node):
                    new_face_id = self.find_containing_face(next_node)
                    
                    if new_face_id is None:
                        continue
                        
                    # Füge die Separatrix hinzu
                    separatrices.append({
                        'coordinates': next_node,
                        'angle': angle,
                        'vector': vec,
                        'singularity_coords': source_node,
                        'face_id': new_face_id
                    })
        
        # Aktualisiere die Separatrizen im Mesh
        self.mesh.separatrices = separatrices
        return separatrices

    def is_boundary_node_regular(self, node_id):
        """Prüft, ob ein Randknoten regulär ist (d.h. die Randkanten mit dem Cross-Vektor übereinstimmen)"""
        tol = 1e-3
        mask_boundaryEdges = self.mesh.edge_attr == 1
        
        boundary_edges = self.mesh.edge_index[:, mask_boundaryEdges]
        boundary_edges_of_node = (torch.where(boundary_edges[0, :] == node_id))[0]
        
        # Wenn der Knoten nicht genau 2 Randkanten hat, ist er nicht regulär
        if len(boundary_edges_of_node) != 2:
            return False
            
        neighbours_idx = boundary_edges[1, boundary_edges_of_node]
        
        source_node = self.mesh.x[node_id, 0:2]
        destination_node0 = self.mesh.x[neighbours_idx[0], 0:2]
        destination_node1 = self.mesh.x[neighbours_idx[1], 0:2]
        
        edge0 = destination_node0 - source_node
        edge1 = destination_node1 - source_node
        edges = [edge0, edge1]
        
        ref_vec = self.mesh.u[node_id]
        base_angle = torch.atan2(ref_vec[1], ref_vec[0])
        if base_angle < 0:
            base_angle = base_angle + 2 * torch.pi
            
        edges_aligns = [False, False]
        
        # Prüfe für jede Kante, ob sie mit einem der Cross-Vektoren übereinstimmt
        for i in range(2):
            edge = edges[i]
            angle_edge = torch.atan2(edge[1], edge[0])
            if angle_edge < 0:
                angle_edge = angle_edge + 2 * torch.pi
                
            for k in range(4):
                angle_cross = base_angle / 4 + k * torch.pi / 2
                if torch.abs(angle_cross - angle_edge) < tol:
                    edges_aligns[i] = True
                    break
                    
        # Der Knoten ist regulär, wenn beide Kanten ausgerichtet sind
        return edges_aligns[0] and edges_aligns[1]

    def get_best_cross_vector(self, point, previous_direction, mesh, containing_face_idx):
        """Ermittelt den besten Cross-Vektor am gegebenen Punkt basierend auf der vorherigen Richtung"""
        if containing_face_idx is None:
            return None, mesh
            
        # Knoten des enthaltenden Dreiecks
        face_indices = mesh.faces[:, containing_face_idx]
        vertices = mesh.x[face_indices, 0:2]
        
        # Referenzvektoren an den Knoten des Dreiecks
        ref_vecs = (mesh.u[face_indices]).to(torch.float)
        
        # Baryzentrische Koordinaten des Punktes berechnen
        try:
            bary_coords = self.compute_barycentric_coordinates(point, vertices)
        except ValueError:
            return None, mesh
            
        # Interpoliere den Referenzvektor am Punkt
        interpolated_vec = torch.einsum('i,ij->j', bary_coords, ref_vecs)
        norm = torch.norm(interpolated_vec)
        
        if norm < 1e-10:
            return None, mesh
            
        interpolated_vec = interpolated_vec / norm  # Normalisierung
        
        # Generiere die vier Cross-Vektoren
        cross_vectors = []
        base_angle = torch.atan2(interpolated_vec[1], interpolated_vec[0]) / 4
        
        for i in range(4):
            angle = base_angle + i * (torch.pi / 2)
            v = torch.tensor([torch.cos(angle), torch.sin(angle)])
            cross_vectors.append(v)
            
        # Wähle den Cross-Vektor, der am besten mit der vorherigen Richtung übereinstimmt
        max_dot = -float('inf')
        best_vector = None
        
        for v in cross_vectors:
            dot = torch.dot(previous_direction, v)
            if dot > max_dot:
                max_dot = dot
                best_vector = v
                
        return best_vector, mesh

    def compute_barycentric_coordinates(self, point, vertices):
        """Berechnet die baryzentrischen Koordinaten eines Punktes in einem Dreieck"""
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = point - vertices[0]
        
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            raise ValueError("Degeneriertes Dreieck.")
            
        invDenom = 1 / denom
        v = (d11 * d20 - d01 * d21) * invDenom
        w = (d00 * d21 - d01 * d20) * invDenom
        u = 1 - v - w
        
        return torch.tensor([u, v, w])

    def find_containing_face(self, point):
        """Findet das Dreieck, das den gegebenen Punkt enthält"""
        nodes = self.mesh.x[:, 0:2]
        node_to_face_ID = self.mesh.nodes_faces_ids
        faces = self.mesh.faces
        
        # Konvertiere zu NumPy für einfachere Handhabung
        if isinstance(point, torch.Tensor):
            point_np = point.detach().numpy()
        else:
            point_np = point
            
        # Finde den nächsten Knoten
        if isinstance(nodes, torch.Tensor):
            nodes_np = nodes.detach().numpy()
        else:
            nodes_np = nodes
            
        distances = np.linalg.norm(nodes_np - point_np, axis=1)
        closest_node_idx = np.argmin(distances)
        
        # Prüfe alle Dreiecke, die mit diesem Knoten verbunden sind
        possible_faces = node_to_face_ID[closest_node_idx]
        
        for face_idx in possible_faces:
            face_vertices = faces[:, face_idx]
            triangle_vertices = nodes[face_vertices, 0:2]
            
            if self.is_point_in_triangle(point, triangle_vertices):
                return face_idx
                
        # Wenn der Punkt nicht in einem der nächstgelegenen Dreiecke liegt, suche weiter
        # Diese erweiterte Suche ist optional und rechenintensiv
        sorted_indices = np.argsort(distances)
        
        for node_idx in sorted_indices[1:10]:  # Prüfe die 10 nächsten Knoten
            if node_idx >= len(node_to_face_ID):
                continue
                
            possible_faces = node_to_face_ID[node_idx]
            
            for face_idx in possible_faces:
                face_vertices = faces[:, face_idx]
                triangle_vertices = nodes[face_vertices, 0:2]
                
                if self.is_point_in_triangle(point, triangle_vertices):
                    return face_idx
                    
        return None

    def is_point_inside_mesh(self, point):
        """Prüft, ob ein Punkt innerhalb des Meshs liegt"""
        face_idx = self.find_containing_face(point)
        return face_idx is not None

    def is_point_in_triangle(self, point, vertices):
        """Prüft, ob ein Punkt innerhalb eines Dreiecks liegt"""
        # Vektoren berechnen
        v0 = vertices[2] - vertices[0]
        v1 = vertices[1] - vertices[0]
        v2 = point - vertices[0]
        
        # Skalarprodukte berechnen
        dot00 = torch.dot(v0, v0)
        dot01 = torch.dot(v0, v1)
        dot02 = torch.dot(v0, v2)
        dot11 = torch.dot(v1, v1)
        dot12 = torch.dot(v1, v2)
        
        # Baryzentrische Koordinaten berechnen
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return False  # Degeneriertes Dreieck
            
        invDenom = 1 / denom
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        
        # Prüfe, ob der Punkt innerhalb des Dreiecks liegt
        # Toleranz hinzufügen für numerische Stabilität
        eps = 1e-7
        return (u >= -eps) and (v >= -eps) and (u + v <= 1 + eps)

    def visualize_separatrices(self, plot_length=0.05):
        """Visualisiert die gefundenen Separatrizen für Debugging"""
        import matplotlib.pyplot as plt
        
        # Knoten des Meshs
        nodes = self.mesh.x[:, 0:2].detach().numpy()
        
        # Faces des Meshs
        faces = self.mesh.faces.detach().numpy()
        
        plt.figure(figsize=(12, 10))
        
        # Plotte das Mesh
        for i in range(faces.shape[1]):
            face = faces[:, i]
            x = nodes[face, 0]
            y = nodes[face, 1]
            plt.plot(np.append(x, x[0]), np.append(y, y[0]), 'k-', linewidth=0.5)
            
        # Plotte Singularitäten
        for face_id, coords in self.mesh.singularities_coords.items():
            sing_type = self.mesh.singularities[face_id].item()
            if sing_type > 0:
                plt.plot(coords[0], coords[1], 'ro', markersize=8)  # Rot für positive Singularitäten
            else:
                plt.plot(coords[0], coords[1], 'bo', markersize=8)  # Blau für negative Singularitäten
                
        # Plotte Separatrizen
        for sep in self.mesh.separatrices:
            start = sep['singularity_coords'].detach().numpy()
            dir_vec = sep['vector'].detach().numpy()
            end = sep['coordinates'].detach().numpy()
            
            # Plotte eine Linie vom Startpunkt in Richtung des Vektors
            plt.arrow(start[0], start[1], dir_vec[0] * plot_length, dir_vec[1] * plot_length, 
                      head_width=0.01, head_length=0.015, fc='green', ec='green')
            
            # Plotte den Punkt auf der Kante
            plt.plot(end[0], end[1], 'g.', markersize=6)
            
        plt.axis('equal')
        plt.title('Mesh mit Singularitäten und Separatrizen')
        plt.grid(True)
        plt.show()
