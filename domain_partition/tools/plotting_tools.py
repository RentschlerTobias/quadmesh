from tools import streamline_simplificator


def plt_faces(mesh):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    nodes = mesh.x
    faces = mesh.faces
    streamline_mapping = mesh.edge_to_streamline_mapping

    num_edges_each_faces = faces.size(0)

    for i, face in enumerate(faces.T):
        streamlines = []
        rand_color = np.random.rand(3,)
        for j in range(num_edges_each_faces):
            edge_idx1 = j
            edge_idx2 = (j + 1) % num_edges_each_faces

            edge = (face[edge_idx1].item(), face[edge_idx2].item())
            streamline = streamline_mapping[edge]
            streamlines.append(streamline)

        surface = np.vstack(streamlines)
        plt.fill(surface[:, 0], surface[:, 1], color=rand_color)


def plot_streamline_dicc(Streamlines, output_file="./figures/streamlines/streamlines_dicc.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.figure(figsize=(5, 5))

    for key in Streamlines.keys():
        streamline = np.array(Streamlines[key]["coords"])  # (N, 2)
        color = np.random.rand(3,)  # random color per streamline
        plt.plot(streamline[:, 0], streamline[:, 1], color=color)

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)
    plt.close()


def plot_streamlines_independet(mesh):

    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = mesh.streamlines
    num_sl = len(streamlines)
    for i in range(num_sl):
        figsize = (5, 5)
        plt.figure(figsize=figsize)
        streamline = mesh.streamlines[i]
        color = np.random.rand(3,)  # Random RGB color for each face

        plt.plot(streamline[:, 0], streamline[:, 1], color=color)
        output_file = f"./figures/streamlines/streamline_{i}.png"
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('equal')  # Equal aspect ratio
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, transparent=True)
        plt.close()


def plot_faces_independet(mesh):

    import matplotlib.pyplot as plt
    import numpy as np

    faces = mesh.quad_faces
    nodes = mesh.quad_coordinates

    num_faces = faces.size(1)
    for i in range(num_faces):
        figsize = (5, 5)
        plt.figure(figsize=figsize)

        face = faces[:, i].T
        coords = nodes[face]  # shape (4, 2)
        color = np.random.rand(3,)  # Random RGB color for each face
        plt.fill(coords[:, 0], coords[:, 1], color=color, edgecolor='gray', linewidth=0.5)

        output_file = f"./figures/faces/face_{i}.png"
        plt.xlim = [0, 1]
        plt.ylim = [0, 1]
        plt.axis('equal')  # Equal aspect ratio
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, transparent=True)
        plt.close()


def plot_post_processed_streamline(streamlines, output_file="./figures/streamlines/streamlines_post_processed.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    for streamline in streamlines:
        color = np.random.rand(3)
        streamline = np.array(streamline)  # Convert to numpy array for easier plotting
        plt.plot(streamline[:, 0], streamline[:, 1], color=color)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_final_mesh(mesh, output_file="./figures/quad_mesh.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    figsize = (5, 5)
    plt.figure(figsize=figsize)
    faces = mesh.quad_faces
    nodes = mesh.quad_coordinates

    for face in faces.T:
        coords = nodes[face]  # shape (4, 2)
        color = np.random.rand(3,)  # Random RGB color for each face
        plt.fill(coords[:, 0], coords[:, 1], color=color, edgecolor='gray', linewidth=0.5)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_block_mesh(block_mesh, output_file="./figures/blocking.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    figsize = (5, 5)
    # Extracting nodes and faces from the block_mesh object
    nodes = block_mesh.x[:, 0:2]  # Assuming block_mesh.x contains node coordinates (n_nodes, 2)
    faces = block_mesh.faces       # Assuming block_mesh.faces is (4, n_faces), 4 corners for each face

    # Extracting edge points and indices
    edge_subdomain_points = block_mesh.edge_subdomain_points
    edge_subdomain_index = block_mesh.edge_subdomain_index

    # Create a plot for the mesh
    plt.figure(figsize=figsize)

    # Plot the faces with random colors
    for face in faces.T:
        coords = nodes[face]  # shape (4, 2)
        color = np.random.rand(3,)  # Random RGB color for each face
        plt.fill(coords[:, 0], coords[:, 1], color=color, edgecolor='gray', linewidth=0.5)

    # Plot the curved edges
    for i, edge_points in enumerate(edge_subdomain_points):
        edge_indices = edge_subdomain_index[:, i]  # Get the indices of the edge
        color = np.random.rand(3,)  # Random color for each edge
        for edge_point in edge_points:
            plt.plot(edge_point[:, 0], edge_point[:, 1], color=color, lw=1)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_domain_partition(mesh, output_file="./figures/streamlines.png", colored=False):
    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = mesh.streamlines
    edges = mesh.edge_index
    # Create a figure for plotting
    plt.figure(figsize=(5, 5))

    for streamline in streamlines:
        streamline = np.array(streamline)  # Convert to numpy array for easier plotting
        if colored == False:
            plt.plot(streamline[:, 0], streamline[:, 1], 'k')
        else:
            color = np.random.rand(3,)  # generate random rgb values
            plt.plot(streamline[:, 0], streamline[:, 1], color=color)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_streamlines(mesh, output_file="./figures/streamlines.png", colored=False):
    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = mesh.streamlines

    # Create a figure for plotting
    plt.figure(figsize=(5, 5))

    # color = 'r'
    for streamline in streamlines:

        color = np.random.rand(3)
        streamline = np.array(streamline)  # Convert to numpy array for easier plotting
        start = streamline[0, :]
        end = streamline[-1, :]
        plt.plot(streamline[:, 0], streamline[:, 1], color=color)

    # plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_intersections(mesh, output_file="./figures/streamline_intersections.png", colored=False):
    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = mesh.streamlines
    edges = mesh.edge_index
    # Create a figure for plotting
    plt.figure(figsize=(5, 5))

    try:
        faces = mesh.triangle_faces.T
        nodes = mesh.triangle_nodes[:, 0:2]  # Assuming we are using 2D coordinates (x, y)

    except Exception as no_quad_mesh:
        faces = mesh.faces.T
        nodes = mesh.x[:, 0:2]  # Assuming we are using 2D coordinates (x, y)

    for face in faces:  # Transposing to iterate through each face
        triangle = nodes[face, :]
        plt.fill(triangle[:, 0], triangle[:, 1], edgecolor='gray', fill=False, linewidth=0.5)

    for streamline in streamlines:
        streamline = np.array(streamline)  # Convert to numpy array for easier plotting
        if colored == False:
            plt.plot(streamline[:, 0], streamline[:, 1], 'r')
        else:
            color = np.random.rand(3,)  # generate random rgb values
            plt.plot(streamline[:, 0], streamline[:, 1], color=color)

    try:

        for i in range(len(mesh.streamline_intersections_points)):
            intersection_coords = mesh.streamline_intersections_points[i]
            plt.plot(intersection_coords[0], intersection_coords[1], 'ok')
    except:
        print('no streamline intersections')
    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_boundary_egdes(mesh, output_file="./figures/mesh_boundary_edges.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(5, 5))
    for i in range(mesh.edge_index.size(1)):
        p1 = mesh.x[mesh.edge_index[0, i]]
        p2 = mesh.x[mesh.edge_index[1, i]]
        if mesh.edge_attr[i] == 1:
            color = 'k'
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='-', color=color)
        # else: color = 'k'
    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_mesh(mesh, output_file="./figures/mesh_boundary_edges.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(5, 5))
    for i in range(mesh.edge_index.size(1)):
        p1 = mesh.x[mesh.edge_index[0, i]]
        p2 = mesh.x[mesh.edge_index[1, i]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='-', color='b')
    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_egdes(mesh, output_file="./figures/edges_random_colored.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(5, 5))
    for i in range(mesh.edge_index.size(1)):
        p1 = mesh.x[mesh.edge_index[0, i]]
        p2 = mesh.x[mesh.edge_index[1, i]]
        if mesh.edge_attr[i] == 1:
            color = 'k'
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='-', color=color)
        # else: color = 'k'
        # Plot faces

    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:, i]
        nodes = mesh.x[face, 0:2]

        plt.fill(nodes[:, 0], nodes[:, 1], color='grey', alpha=0.1)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_nodes(mesh, output_file="./figures/mesh_nodes.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(mesh.x.size(0)):
        node = mesh.x[i, :].numpy()
        if node[2] == 0:
            color = 'r'
        elif node[2] == 1:
            color = 'b'
        else:
            color = 'k'

        plt.scatter(node[0], node[1], marker='o', color=color)
    plt.savefig(output_file, dpi=300)


def plot_singularities(mesh, output_file="./figures/mesh_with_sing.png"):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))

    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:, i]
        nodes = mesh.x[face, 0:2]

        plt.fill(nodes[:, 0], nodes[:, 1], color='grey', alpha=0.1)

    coords = mesh.x[:, 0:2]
    vectors = mesh.u

    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
    vectors = vectors.cpu().numpy() if hasattr(vectors, "cpu") else vectors

    x, y = coords[:, 0], coords[:, 1]
    u, v = vectors[:, 0], vectors[:, 1]

    plt.quiver(x, y, u, v, angles='xy', scale_units='width',
               scale=50, color='blue', alpha=0.6)

    for key in mesh.singularities_coords:
        coords_sing = mesh.singularities_coords[key]

        plt.plot(coords_sing[0], coords_sing[1], 'or')

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def plot_vector_field(mesh, init=False, output_file="./figures/vector_field.png", face_color=None):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(5, 5))

    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:, i]
        nodes = mesh.x[face, 0:2]

        plt.fill(nodes[:, 0], nodes[:, 1], color='grey', alpha=0.1)

    if init:
        mask = mesh.x[:, 2] != 2  # Mask out specific nodes
        coords = mesh.x[mask, 0:2]
        vectors = mesh.frame_field_coords[mask, :]
    else:
        coords = mesh.x[:, 0:2]
        vectors = mesh.u

      # Ensure numpy format for plotting
    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
    vectors = vectors.cpu().numpy() if hasattr(vectors, "cpu") else vectors

    # Unpack coordinates and vectors
    x, y = coords[:, 0], coords[:, 1]
    u, v = vectors[:, 0], vectors[:, 1]

    # Plot the vector field
    plt.quiver(x, y, u, v, angles='xy', scale_units='width',
               scale=50, color='blue', alpha=0.8)

    plt.axis('off')
    plt.axis('equal')  # Ensure equal scaling for x and y
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, transparent=True)
    # plt.close()


def plot_faces(mesh, output_file="./figures/faces.png", colored=None):
    import matplotlib.pyplot as plt
    import numpy as np

    nodes = mesh.x[:, 0:2]
    plt.figure(figsize=(8, 8))
    for face_ids in mesh.faces.T:
        face = nodes[face_ids, :]
        if colored:
            face_color = np.random.rand(3,)  # generate random rgb values
        else:
            face_color = "none"
        plt.fill(face[:, 0], face[:, 1], color=face_color, edgecolor='gray', linewidth=0.5)

    plt.axis('off')
    plt.axis('equal')  # Ensure equal scaling for x and y
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, transparent=True)


def plot_one_separatrices(mesh, ID, output_file="./figures/separatrix.png"):

    from matplotlib.lines import Line2D  # For custom legend entries
    import matplotlib.pyplot as plt
    import numpy as np
    face_id = mesh.separatrices[ID]['face_id']
    plt.figure(figsize=(6, 6))

    for i in range(len(mesh.separatrices)):

        separatix = mesh.separatrices[i]
        if face_id != separatix['face_id']:
            break
        coords    = mesh.separatrices[i]['coordinates']
        face      = mesh.faces[:, separatix['face_id']]
        nodes     = mesh.x[face, 0:2]

        ref_vec = mesh.u[face, :]

        singularity_coords = separatix['singularity_coords']

        plt.fill(nodes[:, 0], nodes[:, 1], edgecolor='gray', fill=None)

        # plot triangles vertices

        plt.plot([singularity_coords[0], coords[0]], [singularity_coords[1], coords[1]], '-k')
        plt.plot(coords[0], coords[1], 'sg')
        plt.plot(singularity_coords[0], singularity_coords[1], 'or')
        old_face_id = separatix['face_id']

        # Compute four cross directions
        angles = np.arctan2(ref_vec[:, 1], ref_vec[:, 0])  # Compute angles of input vectors

        for k in range(4):  # Loop to plot four symmetric directions
            cross_angles = angles / 4 + k * np.pi / 2  # Rotate by 90 degrees for cross field
            u = np.cos(cross_angles)  # Compute rotated vectors
            v = np.sin(cross_angles)

            # Unpack coordinates
            x, y = nodes[:, 0], nodes[:, 1]

            plt.quiver(x, y, u, v, angles='xy', scale_units='inches',
                       scale=2, color='blue', alpha=0.8, width=0.01)

        for k in range(4):  # Loop to plot four symmetric directions
            # cross_angles = separatix['angle'] + k * np.pi / 2  # Rotate by 90 degrees for cross field
            cross_angles = np.arctan2(coords[1] - singularity_coords[1], coords[0] - singularity_coords[0])
            u = np.cos(cross_angles)  # Compute rotated vectors
            v = np.sin(cross_angles)

            # Unpack coordinates
            x, y = coords[0], coords[1]

            plt.quiver(x, y, u, v, angles='xy', scale_units='inches',
                       scale=2, color='red', alpha=0.8, width=0.01)

        for j in range(3):
            plt.plot(nodes[j, 0], nodes[j, 1], 'ok')

    legend_elements = [

        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label="Singularity"),

        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label="$P_{i/j}$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label="P"),
        Line2D([0], [0], color='red', linewidth=2, marker='>', markersize=6, label=r"$u(P)$"),

        Line2D([0], [0], color='blue', linewidth=2, marker='>', markersize=6, label=r"$v_k$"),
        Line2D([0], [0], color='k', linewidth=2, markersize=6, label=r"Separatrices"),

    ]

    plt.legend(handles=legend_elements,
               loc='upper left',
               #            bbox_to_anchor=(1.05, 1),  # x, y in axes coordinates
               fontsize=12)

    # plt.xlim([0.755,0.786])
    # plt.ylim([0.442,0.474])
    #
    # plt.gca().set_aspect('equal', adjustable='box')
#     plt.grid(True)
    # plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='svg')
# def plot_streamlines(mesh,output_file="./figures/domain_partition.png"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     streamlines = mesh.streamlines
#     nodes = mesh.x[:, 0:2]  # Assuming we are using 2D coordinates (x, y)
#     edges = mesh.edge_index
#     # Create a figure for plotting
#     plt.figure(figsize=(8, 8))
#
#     # Plot the mesh by drawing the triangles
#     for face in mesh.faces.T:  # Transposing to iterate through each face
#         triangle = nodes[face, :]
#         plt.fill(triangle[:, 0], triangle[:, 1], edgecolor='gray', fill=False, linewidth=0.5)
#
#     for streamline in streamlines:
#         streamline = np.array(streamline)  # Convert to numpy array for easier plotting
#         plt.plot(streamline[:, 0], streamline[:, 1],'r')
#
#     # Set plot details
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Streamlines over Mesh')
#     plt.gca().set_aspect('equal', adjustable='box')
# #     plt.legend()
#     plt.grid(True)
#
#     plt.savefig(output_file, dpi=300)
#


def plot_cross_field(mesh, init=True, output_file="./figures/cross_field.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    plt.figure(figsize=(5, 5))

    # Select nodes and vectors based on `init` flag
    if init:
        mask = mesh.x[:, 2] != 2  # Mask out specific nodes
        coords = mesh.x[mask, 0:2]
        vectors = mesh.u[mask, :]
    else:
        coords = mesh.x[:, 0:2]
        vectors = mesh.u

    # Convert to numpy if tensor
    coords = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    vectors = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else vectors

    # Compute angles of vectors
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Loop to plot four symmetric cross directions
    for k in range(4):
        cross_angles = angles / 4 + k * np.pi / 2  # Rotate by 90 degrees
        u = np.cos(cross_angles)
        v = np.sin(cross_angles)

        # Unpack coordinates
        x, y = coords[:, 0], coords[:, 1]

        plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                   scale=35, color='blue', alpha=0.8, width=0.004)

    # Plot edges
    edges = mesh.edge_index
    for e in range(edges.size(1)):
        if mesh.edge_attr[e] == 1:  # Only plot edges with attribute 1
            n1 = mesh.x[edges[0, e], :]
            n2 = mesh.x[edges[1, e], :]
            plt.plot([n1[0], n2[0]], [n1[1], n2[1]], 'k', alpha=0.5)

    # Plot faces
    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:, i]
        nodes = mesh.x[face, 0:2]

        plt.fill(nodes[:, 0], nodes[:, 1], color='grey', alpha=0.1)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)
