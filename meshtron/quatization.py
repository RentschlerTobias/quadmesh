import torch
from dataset import MeshData
from tokenizer import Tokenizer2D

min_edge_length = float('inf')
# path = '../data/quad_data.pt'
path = '../data/unstructured_quad_meshes_v2.pt'

meshes = torch.load(path0)
meshes = torch.load(path2)


for i, mesh in enumerate(meshes):
    # vertices: [N_vertices, 2] (x, y Koordinaten)
    vertices = mesh.x[:, 0:2]
    if torch_geometric.utils.isolated.contains_isolated_nodes(mesh.edge_index, mesh.x.size(0)):
        print('isolated')

    # quad_faces: [N_faces, 4] (Indizes der 4 Vertices pro Quad)
    quad_faces = mesh.faces.T
    # if quad_faces.size(1) > 220 or quad_faces.size(1) < 200:
    #     continue
    #
    # 1. Sammle die 4 Vertices jedes Quads: [N_faces, 4, 2]
    # Beispiel: quad_coords[i, 0, :] ist der erste Vertex des i-ten Quads
    quad_coords = vertices[quad_faces]

    # 2. Extrahiere die 4 Kantenvektoren pro Quad
    # Quad V0 -> V1, V1 -> V2, V2 -> V3, V3 -> V0

    # Kante 1: V1 - V0 (Index 1 minus Index 0)
    edge_1 = quad_coords[:, 1, :] - quad_coords[:, 0, :]

    # Kante 2: V2 - V1
    edge_2 = quad_coords[:, 2, :] - quad_coords[:, 1, :]

    # Kante 3: V3 - V2
    edge_3 = quad_coords[:, 3, :] - quad_coords[:, 2, :]

    # Kante 4: V0 - V3 (Zurück zum Start)
    edge_4 = quad_coords[:, 0, :] - quad_coords[:, 3, :]

    # 3. Berechne die Längen der Vektoren (Euklidische Norm / L2-Norm)
    # Längen: [N_faces]

    # Norm-Berechnung: sqrt(dx^2 + dy^2)
    length_1 = torch.norm(edge_1, dim=1)
    length_2 = torch.norm(edge_2, dim=1)
    length_3 = torch.norm(edge_3, dim=1)
    length_4 = torch.norm(edge_4, dim=1)

    # 4. Finde die kleinste Kantenlänge in diesem Mesh

    # Alle Längen in einem Tensor zusammenfassen: [N_faces * 4]
    all_lengths = torch.cat([length_1, length_2, length_3, length_4])

    # Kleinste Kante im aktuellen Mesh
    min_mesh_length = torch.min(all_lengths).item()

    # 5. Aktualisiere die globale kleinste Kantenlänge
    min_edge_length = min(min_edge_length, min_mesh_length)
    if min_edge_length == 0:
        print(f'0 edege in i: {i}')
        break
print(f"Die kleinste Kante im gesamten Datensatz ist: {min_edge_length}")

print(f"The least quantization is : {2/min_edge_length}")
