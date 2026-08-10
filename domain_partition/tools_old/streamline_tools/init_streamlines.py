import torch
from torch_geometric.data import Data

def init_streamlines(mesh):
    
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

        ref_vecs = mesh.frame_field[current_face, :]
        ref_vecs = ref_vecs.to(torch.float)

        singularity_coords = estimate_singularity_location(nodes_of_current_faces, ref_vecs)
    
        num_of_ts = 10000
        t = torch.arange(0, 1, 1 / num_of_ts)

        num_nodes_triangle = 3
        
        for edge_id in range(num_nodes_triangle):

            if edge_id == num_nodes_triangle - 1:
                edge_vec = nodes_of_current_faces[0, :] - nodes_of_current_faces[edge_id, :]
                ref_vec0 = ref_vecs[edge_id]
                ref_vec1 = ref_vecs[0]
            else:
                edge_vec = nodes_of_current_faces[edge_id + 1, :] - nodes_of_current_faces[edge_id, :]
                ref_vec0 = ref_vecs[edge_id]
                ref_vec1 = ref_vecs[edge_id + 1]

            for i in range(num_of_ts):
                current_node = nodes_of_current_faces[edge_id, :] + t[i] * edge_vec
               
                # Use get_best_cross_vector to interpolate the vector at the current_node
                previous_direction = current_node - singularity_coords
                best_vector, _ = get_best_cross_vector(current_node, previous_direction, mesh, face_id)

                separatrix_vec = current_node - singularity_coords
                separatrix_angle = torch.atan2(separatrix_vec[1], separatrix_vec[0]) % (2 * torch.pi)
                cross_vec_angle = torch.atan2(best_vector[1], best_vector[0]) % (2 * torch.pi)
                
                diff_angles = torch.abs(separatrix_angle - cross_vec_angle)

                if diff_angles < tolerance:
                    # Check if a separatrix with a similar angle already exists
                    if already_found_separatrix(separatrices, separatrix_angle, angle_tolerance):
                        update_if_better_separatrix(separatrices, separatrix_angle, current_node, best_vector, angle_tolerance, singularity_coords, face_id)
                    else:
                        separatrices.append({
                            'coordinates': current_node,
                            'angle': separatrix_angle,
                            'vector': best_vector,
                            'singularity_coords': singularity_coords,
                            'face_id': face_id
                        })

    mesh.separatrices = separatrices
    return mesh


def estimate_singularity_location(nodes_of_triangle, ref_vecs, tol=1e-6, max_iters=100):
    # Start with an initial guess for the singularity location (centroid of the triangle)
    singularity_coords = torch.mean(nodes_of_triangle, dim=0)
    
    for _ in range(max_iters):
        # Get the interpolated vector field at the current singularity guess
        interp_vec, u, v, w = interpolate_vector_at_point(singularity_coords, nodes_of_triangle, ref_vecs)
        
        # If the vector magnitude is below the tolerance, return the current guess
        if torch.norm(interp_vec) < tol:
            break
        
        # Move the guess in the opposite direction of the interpolated vector (gradient descent step)
        singularity_coords = singularity_coords - 0.01 * interp_vec
        
        # Ensure singularity_coords stays within the triangle by clamping the barycentric coordinates
        u, v, w = clamp_barycentric_coords(u, v, w)
        
        # Recalculate the singularity coordinates based on clamped barycentric coordinates
        singularity_coords = u * nodes_of_triangle[0, :] + v * nodes_of_triangle[1, :] + w * nodes_of_triangle[2, :]
    
    return singularity_coords

def interpolate_vector_at_point(point, nodes_of_triangle, ref_vecs):
    # Compute the barycentric coordinates for the point relative to the triangle nodes
    u, v, w = compute_barycentric_coords(point, nodes_of_triangle[0, :], nodes_of_triangle[1, :], nodes_of_triangle[2, :])
    
    # Clamp the barycentric coordinates to ensure the point stays within the triangle
    u, v, w = clamp_barycentric_coords(u, v, w)
    
    # Interpolate the vector field based on the barycentric coordinates
    interpolated_vec = u * ref_vecs[0, :] + v * ref_vecs[1, :] + w * ref_vecs[2, :]
    
    return interpolated_vec, u, v, w


def compute_barycentric_coords(p, A, B, C):
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def clamp_barycentric_coords(u, v, w):
    # Clamping the barycentric coordinates to ensure they remain between 0 and 1
    u = torch.clamp(u, 0, 1)
    v = torch.clamp(v, 0, 1)
    w = torch.clamp(w, 0, 1)

    # Ensure they sum to 1 by renormalizing
    sum_coords = u + v + w
    if sum_coords > 0:
        u /= sum_coords
        v /= sum_coords
        w /= sum_coords
    return u, v, w
