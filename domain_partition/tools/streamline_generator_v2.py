
import torch
import numpy as np
from tools.separatrix_generator import SeparatrixGenerator

class StreamlineGenerator_v2:
    def __init__(self,mesh):
        self.separatrices = SeparatrixGenerator(mesh)

        self.termination_node_range = 10e-3

        self.mesh=self.separatrices.mesh
        self.mesh = self.get_streamlines() 

    def get_streamlines(self):
        
        mesh = self.mesh
        streamlines  = mesh.streamlines
        mesh = self.add_face_streamline_labels(mesh)
        
        mask_c0_nodes = mesh.x[:, 2] == 0
        c0_nodes = mesh.x[mask_c0_nodes, 0:2]
        singularity_coords=torch.tensor([mesh.singularities_coords[sing] for sing in mesh.singularities_coords] )
        
        self.streamline_termination_nodes = torch.cat((singularity_coords,c0_nodes),0)
      
        for i in range(len(mesh.separatrices)):
           
            streamline = []
            start_coords         = (mesh.separatrices[i]['coordinates']).to(torch.float)
            start_direction      = (mesh.separatrices[i]['vector']).to(torch.float)
            singularity_coords   = mesh.separatrices[i]['singularity_coords']
            if  start_direction is None:
                print('start_direction is None',start_direction) 
            streamline.append(singularity_coords.numpy())
            streamline.append(start_coords.numpy())
            streamline = self.runge_kutta_heun_integrate_streamline(start_coords,start_direction,mesh,streamline)
            streamlines.append(streamline)
        mesh.streamlines = streamlines

        return mesh

    def add_face_streamline_labels(self,mesh):
        num_faces = mesh.faces.size(1)
        face_streamline_labels = torch.zeros((num_faces))
        mesh.face_streamline_labels = face_streamline_labels
        return mesh

    def find_containing_face(self,point, mesh):
        
        nodes           = mesh.x[:,0:2]
        node_to_face_ID = mesh.nodes_faces_ids
        faces           = mesh.faces
        
        distances        = np.linalg.norm(nodes - point, axis=1)
        closest_node_idx = np.argmin(distances)
        possible_faces   = node_to_face_ID[closest_node_idx]
        
        for face_idx in possible_faces:
            face_vertices = faces[:,face_idx]
            triangle_vertices = nodes[face_vertices,0:2]
            if self.is_point_in_triangle(point, triangle_vertices):
                return face_idx

        return None

    def is_point_inside_mesh(self,point, mesh):
        face_idx = self.find_containing_face(point, mesh)
        return face_idx is not None

    def is_point_in_triangle(self,point, vertices):
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


    def check_termination_criteria(self, point, face_idx, origin):
        
        if face_idx == None:
            return True, point
       
        distances = torch.linalg.norm(self.streamline_termination_nodes - point, dim=1)
        min_distance = torch.min(distances)
        
        if min_distance < self.termination_node_range:
            idx_termination_node = torch.where(distances == min_distance)[0]
            termination_node = self.streamline_termination_nodes[idx_termination_node.item(), :]
            if torch.linalg.norm(point-origin) < self.termination_node_range:
               # Is directly at the start
               return False, None
            else:
                return True, termination_node
               
        return False, None

    def runge_kutta_heun_integrate_streamline(self,start_point, start_direction, mesh, streamline, step_size=None):
       
        if step_size == None:
            step_size = self.termination_node_range/3
        current_point = start_point.clone().detach()
        current_direction = start_direction / torch.norm(start_direction)  # Ensure unit vector
        current_face_idx = self.find_containing_face(current_point, mesh)
        init_face_idx = current_face_idx
        mesh.face_streamline_labels[current_face_idx] = 1

        while True:           
            # Evaluate the vector field at the current point
            v_current,mesh = self.get_best_cross_vector(current_point, current_direction, mesh, current_face_idx)
            if v_current is None:
                print(f"Current point: {current_point} of face {current_face_idx}")
                break
            v_current = v_current / torch.norm(v_current)  # Ensure unit vector

            predictor_point   = current_point + step_size * v_current
            predicted_face_id = self.find_containing_face(predictor_point, mesh) 

            terminate, end_point = self.check_termination_criteria(predictor_point, predicted_face_id, start_point)   
            if terminate == True:
                streamline.append(end_point.numpy())
                break

            if predicted_face_id != current_face_idx:
                mesh.face_streamline_labels[predicted_face_id] = 1

            v_predictor,mesh = self.get_best_cross_vector(predictor_point, v_current, mesh,predicted_face_id)

            if v_predictor is None:
                print('no interpolated Vector at predicted point found')
                break

            v_predictor = v_predictor / torch.norm(v_predictor)  # Ensure unit vector

            average_direction = (v_current + v_predictor) / 2.0
            average_direction = average_direction / torch.norm(average_direction)  # Normalize

            next_point        = current_point + step_size * average_direction
            new_face_id       = self.find_containing_face(next_point, mesh) 
            
            terminate, end_point = self.check_termination_criteria(next_point, new_face_id, start_point)   
            if terminate == True:
                streamline.append(end_point.numpy())
                break
            if  new_face_id != predicted_face_id:
                mesh.face_streamline_labels[new_face_id] = 1
                    
            # Update current point and direction
            current_point = next_point
            current_direction = average_direction
            current_face_idx = new_face_id
            streamline.append(current_point.numpy())

        return np.array(streamline)

    def get_best_cross_vector(self, point, previous_direction, mesh,containing_face_idx):


        if containing_face_idx is None:
            print(f'point ({point}) is not inside a face')
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

    def compute_barycentric_coordinates(self,point, vertices):
        # Computes the barycentric coordinates of a point with respect to a triangle.
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = point - vertices[0]

        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)

        # Compute denominators
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            raise ValueError("Degenerate triangle.")

        invDenom = 1 / denom
        v = (d11 * d20 - d01 * d21) * invDenom
        w = (d00 * d21 - d01 * d20) * invDenom
        u = 1 - v - w

        return torch.tensor([u, v, w])
