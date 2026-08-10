def plot_egdes(mesh, output_file="./figures/mesh_boundary_edges.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(mesh.edge_index.size(1)):
        p1 = mesh.x[mesh.edge_index[0,i]]
        p2 = mesh.x[mesh.edge_index[1,i]]
        if mesh.edge_attr[i] == 1:
            color = 'r'
            plt.plot([p1[0],p2[0]], [p1[1],p2[1]], linestyle='-', color=color)
        else: color = 'k'
    plt.savefig(output_file, dpi=300)

def plot_nodes(mesh, output_file="./figures/mesh_nodes.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(mesh.x.size(0)):
        node = mesh.x[i,:].numpy() 
        if node[2] == 0:
            color = 'r'
        elif node[2]==1:
            color = 'b'
        else: color = 'k'
    
        plt.scatter(node[0], node[1],marker = 'o', color=color)
    plt.savefig(output_file, dpi=300)


def plot_singularities(mesh,output_file = "./figures/mesh_with_sing.png"):

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:,i]
        nodes = mesh.x[face,0:2]
        if mesh.singularities[i]==0:
            color = 'grey'
        elif mesh.singularities[i] == -1:
            color = 'blue'
        elif mesh.singularities[i] == 1:
            color ='red'
        else:
            color = 'black'
        plt.fill(nodes[:,0],nodes[:,1],color = color,alpha = 0.5)

    plt.savefig(output_file, dpi=300)

def plot_vector_field(mesh, init = False, output_file="./figures/vector_field.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract 2D node coordinates and vectors
    coords = mesh.x[:, 0:2]
    
    if init == False:
        vectors = mesh.u
    else:
        vectors = mesh.frame_field_coords

    # Ensure numpy format for plotting
    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
    vectors = vectors.cpu().numpy() if hasattr(vectors, "cpu") else vectors

    # Unpack coordinates and vectors
    x, y = coords[:, 0], coords[:, 1]
    u, v = vectors[:, 0], vectors[:, 1]

    # Plot the vector field
    plt.figure(figsize=(5, 5))
    plt.quiver(x, y, u, v, angles='xy', scale_units='width', 
               scale=50, color='blue', alpha=0.8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Vector Field")
    plt.axis('equal')  # Ensure equal scaling for x and y
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    #plt.close()


import matplotlib.pyplot as plt

def plot_streamlines(mesh,output_file="./figures/domain_partition.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = mesh.streamlines
    nodes = mesh.x[:, 0:2]  # Assuming we are using 2D coordinates (x, y)
    edges = mesh.edge_index
    # Create a figure for plotting
    plt.figure(figsize=(8, 8))
    
    # Plot the mesh by drawing the triangles
    for face in mesh.faces.T:  # Transposing to iterate through each face
        triangle = nodes[face, :]
        plt.fill(triangle[:, 0], triangle[:, 1], edgecolor='gray', fill=False, linewidth=0.5)
    for e in range(edges.size(1)):
        if mesh.edge_attr[e]==1:
            n1 = mesh.x[edges[0,e],:]
            n2 = mesh.x[edges[1,e],:]
            plt.plot([n1[0],n2[0]],[n1[1],n2[1]],'r')
    # Plot each streamline
    for streamline in streamlines:
        streamline = np.array(streamline)  # Convert to numpy array for easier plotting
        plt.plot(streamline[:, 0], streamline[:, 1],'r')
    
    # Set plot details
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Streamlines over Mesh')
    plt.gca().set_aspect('equal', adjustable='box')
#     plt.legend()
    plt.grid(True)
    
    plt.savefig(output_file, dpi=300)

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
    plt.savefig(output_file, dpi=300)
