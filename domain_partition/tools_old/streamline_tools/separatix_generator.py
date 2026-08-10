def get_separatrices_from_singularity(mesh):
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

        singularity_coords = get_singularities_coords(mesh, face_id)
    
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

