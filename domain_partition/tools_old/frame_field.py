import torch
import numpy as np
from tools.mesh_generator import MeshGenerator
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FrameField:
    def __init__(self, meshOfMeshGenerator):

        self.mesh = meshOfMeshGenerator
        self.add_cross_at_boundaries()
        self.generate_cross_field()

    def map_cross_vectors_to_reference_vector(self, angle_rad):
        pi = torch.tensor(math.pi)

        if angle_rad < 0:
            angle_rad = angle_rad + 2*pi
        angle = angle_rad % (pi/2)
        angles = torch.tensor(
            [angle, angle + (pi/2), angle + (pi), angle + (3/2*pi)])
        ref_vec_of_cross = 4*torch.min(angles)
        return ref_vec_of_cross

    def add_cross_at_boundaries(self):

        num_nodes = self.mesh.x.size(0)
        mask_boundaryNodes = self.mesh.x[:, 2] != 2

        mask_boundaryEdges = self.mesh.edge_attr == 1
        boundary_edges = self.mesh.edge_index[:, mask_boundaryEdges[0]]
        midpoint = torch.mean(self.mesh.x[:, :2], dim=0)

        frame_field_angle = torch.zeros((num_nodes), dtype=torch.float)
        frame_field_coords = torch.zeros((num_nodes, 2), dtype=torch.float)
        normals = torch.zeros((num_nodes, 2), dtype=torch.float)

        for i in range(num_nodes):
            if mask_boundaryNodes[i] == True:
                boundary_edges_of_node = (
                    torch.where(boundary_edges[0, :] == i))[0]
                neighbours_idx = boundary_edges[1, boundary_edges_of_node]

                source_node = self.mesh.x[i, 0:2]
                destination_node0 = self.mesh.x[neighbours_idx[0], 0:2]
                destination_node1 = self.mesh.x[neighbours_idx[1], 0:2]
                edge0 = destination_node0 - source_node
                edge1 = destination_node1 - source_node

                #                length0           = torch.sqrt(torch.pow(destination_node0[0]-source_node[0],2)+torch.pow(destination_node0[1]-source_node[1],2))
                #                length1           = torch.sqrt(torch.pow(destination_node1[0]-source_node[0],2)+torch.pow(destination_node1[1]-source_node[1],2))
                #                length            = (length0+length1)/2
                #                length            = length0
                edge_midPoint = midpoint - source_node            # Calculate the angle
                # Convert radians to degrees and adjust to the range 0 to 360
                angle0 = (torch.atan2(edge0[1], edge0[0]))
                # Convert radians to degrees and adjust to the range 0 to 360
                angle1 = (torch.atan2(edge1[1], edge1[0]))

                angle_midpoint = (torch.atan2(
                    edge_midPoint[1], edge_midPoint[0]))

                angle = (angle0)  # +angle1) /2

                diff_angle = torch.absolute(angle - angle_midpoint)
                if diff_angle < torch.tensor(np.pi/2):
                    if angle > torch.tensor(np.pi):

                        angle -= torch.tensor(np.pi)
                    else:
                        angle += torch.tensor(np.pi)
                if angle0 < 0:
                    angle0 += torch.tensor(2*np.pi)

                if angle1 < 0:
                    angle1 += torch.tensor(2*np.pi)

                angle_ref_cross_vec = self.map_cross_vectors_to_reference_vector(
                    angle)

                frame_field_angle[i] = angle_ref_cross_vec
                frame_field_coords[i, 0] = torch.cos(angle_ref_cross_vec)
                frame_field_coords[i, 1] = torch.sin(angle_ref_cross_vec)
            else:
                frame_field_angle[i] = 0
                frame_field_coords[i, 0] = 0
                frame_field_coords[i, 1] = 0

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
        interior_nodes_indices = np.where(self.mesh.x[:, 2] == 2)[0]
        boundary_nodes_indices = np.where(self.mesh.x[:, 2] != 2)[0]
        num_dofs = num_nodes * 2  # Anzahl der Freiheitsgrade (2 pro Knoten)
        A = np.zeros((num_dofs, num_dofs))
        b = np.zeros(num_dofs)
        for e in range(num_elements):

            nodes_indices = elements[e]
            coords = nodes[nodes_indices]  # Shape: (3, 2)
            A_e = self.compute_local_stiffness_matrix(coords)
            b_e = np.zeros((6,))

            # Globale Indizes für die Freiheitsgrade (2 pro Knoten)
            dof_indices = np.zeros(6, dtype=int)
            for i in range(3):
                dof_indices[2*i] = 2 * nodes_indices[i]      # x-Komponente
                dof_indices[2*i+1] = 2 * nodes_indices[i] + 1  # y-Komponente

            for i_local in range(6):
                b[dof_indices[i_local]] += b_e[i_local]
                for j_local in range(6):
                    A[dof_indices[i_local], dof_indices[j_local]
                      ] += A_e[i_local, j_local]

        for idx in boundary_nodes_indices:
            # Indizes der Freiheitsgrade für diesen Knoten
            dof_x = 2 * idx
            dof_y = 2 * idx + 1
            # Setzen der entsprechenden Zeilen in A auf Null und Diagonalelemente auf 1
            A[dof_x, :] = 0
            A[dof_x, dof_x] = 1
            A[dof_y, :] = 0
            A[dof_y, dof_y] = 1
            # Setzen der Werte in b entsprechend den Randbedingungen
            b[dof_x] = self.mesh.frame_field_coords[idx, 0]
            b[dof_y] = self.mesh.frame_field_coords[idx, 1]

        A_sparse = csr_matrix(A)
        u = spsolve(A_sparse, b)

        return A, b, u

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

    # Maximale Anzahl von Iterationen und Toleranz für die Konvergenz
    def Linearization_Norm_Constraint(self, A, b, u_init):

        max_iterations = 100
        tolerance = 1e-6

        # Initialisierung der aktuellen Lösung
        u_current = u_init.copy()

        for n in range(max_iterations):
            # Speichere die vorherige Lösung
            #         print(n)
            u_previous = u_current

            # Schritt a: Linearisierung der Normbedingung
            # Für jeden Knoten formulieren wir die lineare Nebenbedingung

            # Anzahl der Knoten
            num_nodes = self.mesh.x.shape[0]
            num_dofs = num_nodes * 2  # Da wir Vektorfeld mit x- und y-Komponenten haben

            # Aufbau der Matrix C für die Nebenbedingungen
            C = np.zeros((num_nodes, num_dofs))
            d = np.ones(num_nodes)  # Rechte Seite der Nebenbedingungen

            for i in range(num_nodes):
                # Indizes der Freiheitsgrade für Knoten i
                dof_x = 2 * i
                dof_y = 2 * i + 1

                # Aktuelle Werte von u an Knoten i
                u_i_x = u_current[dof_x]
                u_i_y = u_current[dof_y]

                # Gradienten der Normbedingung nach u_x und u_y
                norm_u_i = np.sqrt(u_i_x**2 + u_i_y**2)
                if norm_u_i == 0:
                    # Vermeiden von Division durch Null
                    norm_u_i = 1e-8

                C[i, dof_x] = u_i_x / norm_u_i
                C[i, dof_y] = u_i_y / norm_u_i
                d[i] = 1  # Da die Norm auf 1 gesetzt werden soll

            # Schritt b: Aufstellen des erweiterten Gleichungssystems
            # Erweiterte Matrix und Vektoren
            # [A  C^T] [u]   = [b]
            # [C   0 ] [λ]     [d]

            # Erstellen der erweiterten Matrix
            KKT_matrix = np.zeros((num_dofs + num_nodes, num_dofs + num_nodes))
            KKT_rhs = np.zeros(num_dofs + num_nodes)

            # Füllen der KKT-Matrix
            KKT_matrix[:num_dofs, :num_dofs] = A
            KKT_matrix[:num_dofs, num_dofs:] = C.T
            KKT_matrix[num_dofs:, :num_dofs] = C
            # Die unteren rechten Ecke ist eine Nullmatrix

            # Füllen des rechten Vektors
            KKT_rhs[:num_dofs] = b
            KKT_rhs[num_dofs:] = d

            # Anwenden der Randbedingungen
            # Hier müssen wir sicherstellen, dass die Dirichlet-Randbedingungen erhalten bleiben
            # Dies kann komplex sein, daher werden wir annehmen, dass die Randbedingungen bereits in A und b berücksichtigt sind

            # Lösen des erweiterten Systems
            solution = np.linalg.solve(KKT_matrix, KKT_rhs)

            # Extrahieren der neuen Lösung und der Lagrange-Multiplikatoren
            u_new = solution[:num_dofs]
            lambdas = solution[num_dofs:]

            # Aktualisieren der aktuellen Lösung
            u_current = u_new.copy()

            # Schritt c: Überprüfung der Konvergenz
            diff = np.linalg.norm(u_current - u_previous)
            if diff < tolerance:
                print(f"Konvergenz erreicht nach {n+1} Iterationen.")
                break
        else:
            print("Maximale Anzahl von Iterationen erreicht.")

        return u_current
