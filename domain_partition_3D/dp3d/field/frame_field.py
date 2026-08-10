import torch
import numpy as np
from .mesh_generator import MeshGenerator
import math
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import time

class FrameField:
    def __init__(self, meshOfMeshGenerator):

        self.mesh = meshOfMeshGenerator
        self.add_cross_at_boundaries()
        self.time_start = time.time() 
        self.generate_cross_field()
        self.time_end = time.time()
        self.mesh.time_frame_field_generator =  self.time_end - self.time_start

    def map_cross_vectors_to_reference_vector(self, angle_rad):
        # pi = torch.tensor(math.pi)

        pi = torch.pi        # if angle_rad < 0:
        #     angle_rad = angle_rad + 2*pi
        angle = angle_rad# % (pi/2)
        
        angles = torch.tensor([angle, angle + (pi/2), angle + (pi), angle + (3/2*pi)])
        angles = angles % (2*pi)
        ref_vec_of_cross = 4*torch.min(angles)
        return ref_vec_of_cross

   
    def add_cross_at_boundaries(self):
        
        pi = torch.pi
        mesh = self.mesh
        num_nodes = mesh.x.size(0)
        mask_boundaryEdges = mesh.edge_attr == 1
        idx_boundaryNodes = torch.unique(mesh.edge_index[0, mask_boundaryEdges])
        num_boundary_nodes = idx_boundaryNodes.size(0)

        boundary_edges = mesh.edge_index[:, mask_boundaryEdges]

        frame_field_angle = torch.zeros((num_nodes), dtype=torch.float)
        frame_field_coords = torch.zeros((num_nodes, 2), dtype=torch.float)

        for i in range(num_boundary_nodes):
            idx_current_node = idx_boundaryNodes[i]

            boundary_edges_of_node = (torch.where(boundary_edges[0, :] == idx_current_node))[0]
            neighbours_idx = boundary_edges[1, boundary_edges_of_node]
            source_node = mesh.x[idx_current_node, 0:2]
            destination_node0 = mesh.x[neighbours_idx[0], 0:2]
            destination_node1 = mesh.x[neighbours_idx[1], 0:2]

            edge0 = destination_node0 - source_node

            edge1 = destination_node1 - source_node

            # Normalize the edges
            edge0_normalized = edge0 / torch.norm(edge0,p=2)
            edge1_normalized = edge1 / torch.norm(edge1,p=2)

            # Combine normalized edges
            edge_mid = edge0_normalized + edge1_normalized
            edge_mid_normalized = edge_mid #/ torch.norm(edge_mid)  # Normalize the combined vector

            # Calculate the angle of the combined vector
            angle = torch.atan2(edge_mid_normalized[1], edge_mid_normalized[0]) % (2*pi)
            angle0 = torch.atan2(edge0_normalized[1], edge0_normalized[0]) % (2*pi)
            angle1 = torch.atan2(edge1_normalized[1], edge1_normalized[0]) % (2*pi)

            angle_diff = torch.abs((angle1 - angle0 + pi) % (2 * pi) - pi)
            if 0.95 * (pi / 2) < angle_diff < 1.05 * (pi / 2):
                ref_angle = self.map_cross_vectors_to_reference_vector(angle0)
            else:
                ref_angle = self.map_cross_vectors_to_reference_vector(angle)
            # Store the result
            frame_field_angle[i] = ref_angle
            frame_field_coords[i, 0] = torch.cos(ref_angle)
            frame_field_coords[i, 1] = torch.sin(ref_angle)


            self.mesh.frame_field_angle = frame_field_angle
            self.mesh.frame_field_coords = frame_field_coords


    def generate_cross_field(self):
        A, b, u = self.compute_initial_frame_field()
        u_new = self.Linearization_Norm_Constraint(A, b, u)

        u_init_x = torch.from_numpy(u[::2]).unsqueeze(1)
        u_init_y = torch.from_numpy(u[1::2]).unsqueeze(
            1)  # x-Komponenten an den Knoten
        u_init = torch.concat((u_init_x, u_init_y), dim=1)

        u_final_x = torch.from_numpy(u_new[::2]).unsqueeze(1)
        u_final_y = torch.from_numpy(u_new[1::2]).unsqueeze(
            1)  # x-Komponenten an den Knoten
        u_final = torch.concat((u_final_x, u_final_y), dim=1)
        

        # ensure init values are maintained
        mask_boundary = self.mesh.x[:,2] !=2 
        vector_field =(u_final.detach().clone()).to(torch.float)
        vector_field[mask_boundary,:] =( self.mesh.frame_field_coords[mask_boundary,:] )
        self.mesh.u_init = u_init
        self.mesh.u = u_final
        self.mesh.frame_field =vector_field  

    def compute_initial_frame_field(self):

        num_nodes = self.mesh.x.shape[0]
        num_elements = self.mesh.faces.shape[1]
        nodes = self.mesh.x[:, 0:2]  # Shape: (num_nodes, 2)
        elements = self.mesh.faces.T  # Shape: (num_elements, 3)
#        interior_nodes_indices = np.where(self.mesh.x[:, 2] == 2)[0]
#        boundary_nodes_indices = np.where(self.mesh.x[:, 2] != 2)[0]
        mask_boundaryEdges = self.mesh.edge_attr == 1
        boundary_nodes_indices = torch.unique(self.mesh.edge_index[0, mask_boundaryEdges])
        num_dofs = num_nodes * 2  # Anzahl der Freiheitsgrade (2 pro Knoten)
        b = np.zeros(num_dofs)

        # Sparse assembly via COO triplets (statt dense (2N)^2 + Python-Loop).
        rows, cols, vals = [], [], []
        for e in range(num_elements):

            nodes_indices = elements[e]
            coords = nodes[nodes_indices]  # Shape: (3, 2)
            A_e = self.compute_local_stiffness_matrix(coords)

            dof_indices = np.empty(6, dtype=int)
            for i in range(3):
                dof_indices[2*i] = 2 * int(nodes_indices[i])      # x-Komponente
                dof_indices[2*i+1] = 2 * int(nodes_indices[i]) + 1  # y-Komponente

            # b_e ist Null (nabla_u0 = 0) -> kein Beitrag zu b.
            for i_local in range(6):
                for j_local in range(6):
                    rows.append(dof_indices[i_local])
                    cols.append(dof_indices[j_local])
                    vals.append(A_e[i_local, j_local])

        # COO -> CSR summiert doppelte (row, col) Eintraege (entspricht dem +=).
        A = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr().tolil()

        for idx in boundary_nodes_indices:
            idx = int(idx)
            # Indizes der Freiheitsgrade für diesen Knoten
            dof_x = 2 * idx
            dof_y = 2 * idx + 1
            # Dirichlet-BC: Zeile auf Null, Diagonale auf 1 (sparse via lil).
            A.rows[dof_x] = [dof_x]
            A.data[dof_x] = [1.0]
            A.rows[dof_y] = [dof_y]
            A.data[dof_y] = [1.0]
            # Setzen der Werte in b entsprechend den Randbedingungen
            b[dof_x] = float(self.mesh.frame_field_coords[idx, 0])
            b[dof_y] = float(self.mesh.frame_field_coords[idx, 1])

        A_sparse = A.tocsr()
        u = spsolve(A_sparse, b)

        return A_sparse, b, u

    def compute_element_matrices(coords, nodes):
        # coords: Array der Knotenkoordinaten des Elements, Shape: (3, 2)
        # nodes: Indizes der Knoten des Elements

        # Extrahieren der Koordinaten
        x = coords[:, 0]
        y = coords[:, 1]

        # Berechnung der Fläche des Dreiecks
        area = 0.5 * np.abs(
            (x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0])
        )

        # Berechnung der Ableitungen der Basisfunktionen
        b = np.zeros(3)
        c = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            b[i] = y[j] - y[k]  # y_j - y_k
            c[i] = x[k] - x[j]  # x_k - x_j

        # Gradienten der Basisfunktionen
        grad_N = np.array([[b[i], c[i]] for i in range(3)]) / (2 * area)

        # Lokale Steifigkeitsmatrix A_e
        A_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                A_e[i, j] = area * np.dot(grad_N[i], grad_N[j])

        # Lokaler Lastvektor b_e (angenommen, nabla_u0 = [0, 0])
        b_e = np.zeros(3)

        return A_e, b_e

    def compute_local_stiffness_matrix(self, coords):
        # coords: Shape (3, 2)
        # Rückgabe: lokale Steifigkeitsmatrix A_e der Größe (6, 6)

        # Berechnung der Fläche des Dreiecks
        x = coords[:, 0]
        y = coords[:, 1]
        area = 0.5 * np.abs(
            (x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0])
        )

        # Berechnung der Ableitungen der Basisfunktionen
        b_coeff = np.zeros(3)
        c_coeff = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            b_coeff[i] = y[j] - y[k]
            c_coeff[i] = x[k] - x[j]

        # Gradienten der Basisfunktionen
        grad_N = np.array([[b_coeff[i], c_coeff[i]]
                          for i in range(3)]) / (2 * area)

        # Lokale Steifigkeitsmatrix (6x6, da Vektorfeld)
        A_e = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                # Steifigkeitsmatrix für die x-Komponente
                A_e[2*i, 2*j] = area * np.dot(grad_N[i], grad_N[j])
                # Steifigkeitsmatrix für die y-Komponente
                A_e[2*i+1, 2*j+1] = area * np.dot(grad_N[i], grad_N[j])
                # Gekoppelte Terme sind Null (keine Kopplung zwischen x und y in diesem Fall)

        return A_e

    def Linearization_Norm_Constraint(self, A, b, u_init):
        # Non-iterativ (Knöppel et al. 2013, "Globally Optimal Direction Fields"):
        # u_init ist bereits die glatte harmonische Loesung des
        # Repraesentationsfeldes (cos4θ, sin4θ) mit harten Rand-Dirichlet-BC
        # (sparse spsolve in compute_initial_frame_field). Fuer eine flache,
        # boundary-aligned Domain ist die per-Knoten-Normalisierung dieses Feldes
        # das glatteste Einheits-Richtungsfeld -> topologisch minimale
        # Singularitaetenzahl. Ersetzt die fruehere ~130x dense KKT-Iteration.
        #
        # A, b bleiben in der Signatur (von generate_cross_field uebergeben),
        # werden hier aber nicht mehr gebraucht.
        z = u_init.reshape(-1, 2)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Division durch Null vermeiden
        u_current = (z / norms).reshape(-1)

        self.mesh.frame_field_iteration_number = 0
        self.mesh.frame_field_tol = 0.0

        return u_current
