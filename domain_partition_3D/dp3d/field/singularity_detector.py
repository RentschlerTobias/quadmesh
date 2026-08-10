import torch
from torch_geometric.data import Data

def detect_singularities(mesh):
    
    num_faces = mesh.faces.size(1)
    
    vec_x = mesh.frame_field[:, 0]
    vec_y = mesh.frame_field[:, 1]
    # Compute angles of the frame field
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

