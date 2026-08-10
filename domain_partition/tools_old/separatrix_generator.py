import torch
import numpy as np
class SeparatrixGenerator:
    def __init__(self,mesh):
        self.mesh = mesh
        self.detect_singularities()
        self.get_separatrices_from_singularity()

        self.get_separatrices_from_c0_nodes()

    def detect_singularities(self):
        mesh = self.mesh 
        num_faces = mesh.faces.size(1)
        
        vec_x = mesh.frame_field[:, 0]
        vec_y = mesh.frame_field[:, 1]
        angles = torch.atan2(vec_y,vec_x)
        
        # Map angles to faces
        angles_of_faces = torch.zeros(3, num_faces)
        angles_of_faces[0, :] = angles[mesh.faces[0, :]]
        angles_of_faces[1, :] = angles[mesh.faces[1, :]]
        angles_of_faces[2, :] = angles[mesh.faces[2, :]]
        
        # Compute normalized angular differences
        angles_of_faces_sum = torch.zeros(3, num_faces)
        angles_of_faces_sum[0, :] = torch.remainder(angles_of_faces[1, :] - angles_of_faces[0, :] + torch.pi , 2 * torch.pi) - torch.pi
        angles_of_faces_sum[1, :] = torch.remainder(angles_of_faces[2, :] - angles_of_faces[1, :] + torch.pi , 2 * torch.pi) - torch.pi
        angles_of_faces_sum[2, :] = torch.remainder(angles_of_faces[0, :] - angles_of_faces[2, :] + torch.pi , 2 * torch.pi) - torch.pi
        
        # Calculate the Poincaré index for each triangle
        poincare_idx = torch.round(torch.sum(angles_of_faces_sum, dim=0) / (2 * torch.pi))
        mesh.singularities = poincare_idx
        
        # Check Singularities:
        poincare_idx_sum = torch.sum(poincare_idx)

        num_nodes = mesh.x.size(0)
        num_edges = mesh.edge_index.size(1)/2
        num_faces = mesh.faces.size(1)
        euler_characteristic = num_nodes - num_edges + num_faces

        if euler_characteristic  !=  poincare_idx_sum :
            print('number of singularities is not correct')
        else:
            print('number of singularities is correct')
        return mesh 



    def get_separatrices_from_singularity(self):

        mesh = self.mesh
        separatrices = []

        num_faces = mesh.faces.size(1)
        face_all_ids = torch.arange(num_faces)
        face_ids = face_all_ids[mesh.singularities != 0]
        
        tolerance = 1e-3
        angle_tolerance = 1e-2
        
        # Iterating over each face that contains a singularity
        for i in range(face_ids.size(0)):
            face_id = face_ids[i]
            current_face = mesh.faces[:, face_id]
            nodes_of_current_faces = mesh.x[current_face, 0:2]

            ref_vecs = mesh.u[current_face, :]
            ref_vecs = ref_vecs.to(torch.float)

            singularity_coords = self.get_singularities_coords(face_id)
        
            num_of_ts = 10000
            t = torch.arange(0, 1, 1 / num_of_ts)

            num_nodes_triangle = 3
            
            for edge_id in range(num_nodes_triangle):

                if edge_id == num_nodes_triangle - 1:
                    edge_vec = nodes_of_current_faces[0, :] - nodes_of_current_faces[edge_id, :]
                else:
                    edge_vec = nodes_of_current_faces[edge_id + 1, :] - nodes_of_current_faces[edge_id, :]

                for i in range(num_of_ts):
                    current_node = nodes_of_current_faces[edge_id, :] + t[i] * edge_vec
                   
                    # Use get_best_cross_vector to interpolate the vector at the current_node
                   
                    separatrix_vec = current_node - singularity_coords
                    best_vector, _ = self.get_best_cross_vector(current_node, separatrix_vec,mesh, face_id)

                    separatrix_angle = torch.atan2(separatrix_vec[1], separatrix_vec[0]) % (2 * torch.pi)
                    cross_vec_angle = torch.atan2(best_vector[1], best_vector[0]) % (2 * torch.pi)
                    
                    diff_angles = torch.abs(separatrix_angle - cross_vec_angle)

                    if diff_angles < tolerance:
                        # Check if a separatrix with a similar angle already exists
                        if self.already_found_separatrix(separatrices, separatrix_angle, angle_tolerance):
                            self.update_if_better_separatrix(separatrices, separatrix_angle, current_node, best_vector, angle_tolerance, singularity_coords, face_id)
                        else:
                            separatrices.append({
                                'coordinates': current_node,
                                'angle': separatrix_angle,
                                'vector': best_vector,
                                'singularity_coords': singularity_coords,
                                'face_id': face_id
                            })

        self.mesh.separatrices = separatrices


    def get_separatrices_from_c0_nodes(self):
        mesh = self.mesh
        mask_c0_nodes = mesh.x[:, 2] == 0
        c0_nodes = mesh.x[mask_c0_nodes, 0:2]
        node_ids = torch.arange(mesh.x.size(0))
        c0_node_ids = node_ids[mask_c0_nodes]
        separatrices = mesh.separatrices
        tolerance = 1e-3
        angle_tolerance = 1e-2
        num_of_ts = 10000
        t = torch.linspace(0, 1, num_of_ts)
        step = 0.001
        for i, node_id_tensor in enumerate(c0_node_ids):
            node_id = node_id_tensor.item()
            
            source_node = mesh.x[node_id, 0:2]
            ref_vec = mesh.u[node_id]
            base_angle = torch.atan2(ref_vec[1], ref_vec[0])
            for k in range(4):
                angle = base_angle/4 +k*torch.pi/2
                vec = torch.tensor([torch.cos(angle), torch.sin(angle)])
                
                next_node = source_node+step*vec
                next_node = next_node.to(torch.float)

                if self.is_point_inside_mesh(next_node):
                    
                    new_face_id = self.find_containing_face(next_node)       
            
                    if new_face_id is None:
                            break
                    separatrices.append({
                                'coordinates': next_node,
                                'angle': angle,
                                'vector': vec,
                                'singularity_coords': source_node,
                                'face_id': new_face_id  # No face since it's a C0 boundary
                            })

        self.mesh.separatrices = separatrices


    def get_best_cross_vector(self,point, previous_direction, mesh,containing_face_idx):


        if containing_face_idx is None:
            return None, mesh  

       
        face_indices = mesh.faces[:, containing_face_idx]
        vertices = mesh.x[face_indices, 0:2]

        # Get the reference vectors at the triangle's vertices
        ref_vecs = (mesh.u[face_indices]).to(torch.float)

        # Compute barycentric coordinates of the point
        bary_coords = self.compute_barycentric_coordinates(point, vertices)

        # Interpolate the reference vector at the point
        interpolated_vec = torch.einsum('i,ij->j', bary_coords, ref_vecs)
        interpolated_vec = interpolated_vec / torch.norm(interpolated_vec)  # Normalize

        # Generate the four cross vectors
        cross_vectors = []
        base_angle = torch.atan2(interpolated_vec[1], interpolated_vec[0])/4
        for i in range(4):
            angle = base_angle + i * (torch.pi / 2)
            v = torch.tensor([torch.cos(angle), torch.sin(angle)])
            cross_vectors.append(v)

        # Select the cross vector that aligns best with the previous direction
        max_dot = -float('inf')
        best_vector = None
        for v in cross_vectors:
            dot = torch.dot(previous_direction, v)
            if dot > max_dot:
                max_dot = dot
                best_vector = v
        return best_vector,mesh

    
    def get_singularities_coords(self,face_id):
        # Computes the singularity location in a triangle using bilinear interpolation.
        # Solves for the point where the interpolated vector field is (0,0).
        face = self.mesh.faces[:, face_id]  
        vectors = self.mesh.u[face]  
        nodes = self.mesh.x[face, 0:2]

        v0, v1, v2 = vectors
        p0, p1, p2 = nodes

        # Define the interpolation equation: V(P) = alpha * v0 + beta * v1 + gamma * v2
        # We solve for (alpha, beta, gamma) such that V(P) = (0, 0) (singularity condition)
        A = torch.stack([v0 - v2, v1 - v2], dim=1)  
        b = -v2  

        try:
            coeffs = torch.linalg.solve(A, b)

            singularity_location = p2 + coeffs[0] * (p0 - p2) + coeffs[1] * (p1 - p2)

        except RuntimeError:
            singularity_location = torch.mean(nodes, dim=0)

        return singularity_location

    def compute_barycentric_coordinates(self,point, vertices):

        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = point - vertices[0]

        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if denom == 0:
            raise ValueError("Degenerate triangle.")

        invDenom = 1 / denom
        v = (d11 * d20 - d01 * d21) * invDenom
        w = (d00 * d21 - d01 * d20) * invDenom
        u = 1 - v - w

        return torch.tensor([u, v, w])


    def already_found_separatrix(self, separatrices, new_angle, angle_tol):

        for entry in separatrices:
            if torch.abs(entry['angle'] - new_angle) < angle_tol:
                return True
        return False

    def update_if_better_separatrix(self,separatrices, new_angle, current_node, cross_vec,angle_tol,singularity_coords,face_id):

        for idx, entry in enumerate(separatrices):
            if torch.abs(entry['angle'] - new_angle) < angle_tol:
                # Keep the one closer to the ideal angle
                if torch.abs(entry['angle'] - new_angle) > torch.abs(new_angle - new_angle):
                    separatrices[idx] = {'coordinates': current_node, 'angle': new_angle, 'vector': cross_vec,'singularity_coords':singularity_coords,'face_id':face_id}  
                return

        # If no match found, add new separatrix
        separatrices.append({'coordinates': current_node, 'angle': new_angle, 'vector': cross_vec,'singularity_coords':singularity_coords,'face_id':face_id})



    def find_containing_face(self,point):
        nodes           = self.mesh.x[:,0:2]
        node_to_face_ID = self.mesh.nodes_faces_ids
        faces           = self.mesh.faces
        
        distances        = np.linalg.norm(nodes - point, axis=1)
        closest_node_idx = np.argmin(distances)
        possible_faces   = node_to_face_ID[closest_node_idx]
        
        for face_idx in possible_faces:
            face_vertices = faces[:,face_idx]
            triangle_vertices = nodes[face_vertices,0:2]
            if self.is_point_in_triangle(point, triangle_vertices):
                return face_idx

        return None

    def is_point_inside_mesh(self,point):

        face_idx = self.find_containing_face(point)
        return face_idx is not None


    def is_point_in_triangle(self,point, vertices):
                # Compute vectors
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
