import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge 
class MeshFromFieldgen:
   
    def __init__(self, obj_path):
        self.mesh = self.parse_obj_to_pytorch_data(obj_path)
        self.mesh.face = self.mesh.faces
        transform = FaceToEdge()
        self.mesh = transform(self.mesh)

    def parse_obj_to_pytorch_data(self, filepath):
        vertices = []
        faces = []
        frame_field_fieldgen = {}
        singularities = {}
        with open(filepath, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith('v '): 
                            _, x, y, z = line.split()
                            vertices.append([float(x), float(y), float(z)])
                        elif line.startswith('f '):  # Face line
                            face_indices = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]
                            faces.append(face_indices)  
                        elif line.startswith('# field '):  # Cross-field vector line
                            _, _, vertex_index, x, y, z = line.split()
                            vertex_index = int(vertex_index) - 1  # Convert to zero-indexed
                            if vertex_index not in frame_field_fieldgen:
                                frame_field_fieldgen[vertex_index] = []
                                frame_field_fieldgen[vertex_index] = [float(x), float(y), float(z)]
                            else:
                                print('second vec')
                        elif line.startswith('# singularity '):  # Singularity line
                            _, _, face_index, degree = line.split()
                            if float(degree) == 0.25:
                                point_care = 1
                            elif float(degree) == -0.25:
                                point_care = -1
                            else:
                                point_care = 2
                            singularities[int(face_index) - 1] = int(point_care)  # Convert to zero-indexed and store degree
           # Convert lists to PyTorch tensors
        coords = torch.tensor(vertices, dtype=torch.float32)  # Node coordinates (x, y, z)
        faces = torch.tensor(faces, dtype=torch.long).t().contiguous()  # Transpose and ensure contiguous
        singularities_tensor = torch.zeros(len(faces[0]), dtype=torch.int) if len(faces) > 0 else torch.zeros(0, dtype=torch.int)
        for face_index, degree in singularities.items():
            singularities_tensor[face_index] = degree

        mapped_frame_field, cross_field_angles,cross_field_vectors = self.get_frame_field_cross_field(frame_field_fieldgen)

        mesh_data = Data(
            x=coords[:, :2],  # Only keep x, y coordinates (since z=0 in 2D)
            faces=faces,  
            cross_field        = cross_field_vectors ,  # Keep only 2D vector components (x, y)
            cross_field_angles = cross_field_angles,
            frame_field        = frame_field_fieldgen,
            singularities      = singularities_tensor
        )
       
        return mesh_data

    def get_frame_field_cross_field(self, frame_field):
        
        # "fieldgen boundary align cross field" produces not a smooth vec field, but returns one of the 4 cross vec, 
        # therefore we mapp it to a smooth frame field and generate the hole 4 cross vecs at each node
        
        frame_field_tensor      = torch.tensor([vectors for vectors in frame_field.values()])  
        angles                  = torch.atan2(frame_field_tensor[:,1],frame_field_tensor[:,0])
        cross_field_angles      = {}
        
        num_nodes = frame_field_tensor.size(0)
        
        mapped_frame_field = torch.zeros((num_nodes,2),dtype =torch.float)
        
        for idx_angle in range(angles.size(0)):
            # calculate angles
            angle = angles[idx_angle]/4
            # 4 orthogonal vecs
            cross =   torch.tensor((angle,angle + 1*torch.pi/2,angle + 2*torch.pi/2,angle + 3*torch.pi/2),dtype = torch.float)
            # make sure angles are in 0 to 2pi and not -pi to pi
            cross = cross %(2*torch.pi)
            # sort the 4 ortho vecs by angle
            sorted_crosses = cross.sort()
            cross = sorted_crosses.values[sorted_crosses.indices]
            
            cross_field_angles[idx_angle] = cross
            mapped_frame_angle = cross[0]*4 
            mapped_frame_field[idx_angle,0] = torch.cos(mapped_frame_angle)
            mapped_frame_field[idx_angle,1] = torch.sin(mapped_frame_angle)
            cross_field_vectors = {
                node: torch.stack((torch.cos(cross_angles), torch.sin(cross_angles)), dim=1) 
                for node, cross_angles in cross_field_angles.items()
            }

        return mapped_frame_field, cross_field_angles,cross_field_vectors


