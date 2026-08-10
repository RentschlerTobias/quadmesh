
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator_v2, StreamlinePostProcessor
from tools import QuadMeshGenerator
from tools import MeshCheck
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import multiprocessing as mp
import time
import tempfile

path = './saved_meshes/new_checkpoints/checkpoint_mesh_10.pt'

meshes = torch.load(path, weights_only=False)
len(meshes)

for i, mesh in enumerate(meshes):
    mesh
    fig_path = f'./figures/mesh_{i}.png'
    plot_final_mesh(mesh, output_file=fig_path)


path = f"./figures/faces/mesh_transfinite_10.png"
plot_final_mesh(mesh, output_file=path)


def extract_mesh_data(tri_mesh, quad_mesh, block_mesh):

    # Extract features of the triangulated mesh incl. frame_field & streamline generation

    tri_coordinates = tri_mesh.x
    tri_faces = tri_mesh.faces
    tri_edges = tri_mesh.edge_index
    tri_edges_attr = tri_mesh.edge_attr
    tri_mesh_face_attr = tri_mesh.face_attr
    streamlines = tri_mesh.streamlines
    frame_field_angle = tri_mesh.frame_field_iteration_number
    frame_field_u = tri_mesh.u
    singularities_coords = tri_mesh.singularities_coords
    frame_field_iteration_number = tri_mesh.frame_field_iteration_number

    frame_field_time = tri_mesh.time_frame_field_generator
    singularities = tri_mesh.singularities

    # Extract the block structure
    blocking_nodes = block_mesh.x
    blocking_faces = block_mesh.faces
    edge_to_streamline = block_mesh.edge_to_streamline
    # Extract transfinite Interpolated mesh
    quad_coordinates = quad_mesh.x
    quad_faces = quad_mesh.faces
    quad_edges = quad_mesh.edge_index

    final_mesh = Data(blocking_nodes=blocking_nodes,
                      blocking_faces=blocking_faces,
                      quad_coordinates=quad_coordinates,
                      quad_faces=quad_faces, quad_edges=quad_edges,
                      singularities=singularities,
                      singularities_coords=singularities_coords,
                      frame_field_time=frame_field_time,
                      frame_field_iteration_number=frame_field_iteration_number,
                      frame_field_u=frame_field_u,
                      frame_field_angle=frame_field_angle,
                      streamlines=streamlines,
                      edge_to_streamline=edge_to_streamline,
                      tri_edges_attr=tri_edges_attr,
                      tri_mesh_face_attr=tri_mesh_face_attr,
                      tri_edges=tri_edges, tri_faces=tri_faces,
                      tri_coordinates=tri_coordinates)

    return final_mesh


success = False
# np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())
while success == False:
    try:
        airfoil = NACA_airfoil()
        random_lc = 0.04 + 0.02 * np.random.rand()
        mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
        frameField = FrameField(mesh_gen.mesh)
        streamline = StreamlineGenerator_v2(frameField.mesh)
        streamlines_post_processed = StreamlinePostProcessor(streamline.mesh)
        blocked_mesh = streamlines_post_processed.block_mesh
        transfiniteInterpolator = QuadMeshGenerator(
            blocked_mesh, transfinite_divisions=10)
        quad_mesh = transfiniteInterpolator.transfinite_mesh
        tri_mesh = streamlines_post_processed.mesh
        mesh_check = MeshCheck(tri_mesh, quad_mesh, tol=2e-3)
        success = mesh_check.is_valid

        print(f'!!! \n area difference: \n {
              mesh_check.quad_area - mesh_check.tri_area}\n !!!')

        mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)
        if success == True:
            mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

    except Exception as e:
        print(f'\n domain partition failed: {e} \n')


transfiniteInterpolator = QuadMeshGenerator(
    blocked_mesh, transfinite_divisions=10)
quad_mesh = transfiniteInterpolator.transfinite_mesh
mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

path = f"./figures/faces/mesh_transfinite_10.png"
plot_final_mesh(mesh, output_file=path)

meta_mesh = mesh

transfiniteInterpolator = QuadMeshGenerator(
    blocked_mesh, transfinite_divisions=3)
quad_mesh = transfiniteInterpolator.transfinite_mesh
mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

path = f"./figures/faces/mesh_transfinite_3.png"
plot_final_mesh(mesh, output_file=path)

meta_mesh.nodes_T3 = mesh.quad_coordinates
meta_mesh.faces_T3 = mesh.quad_faces


transfiniteInterpolator = QuadMeshGenerator(
    blocked_mesh, transfinite_divisions=4)
quad_mesh = transfiniteInterpolator.transfinite_mesh
mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)


path = f"./figures/faces/mesh_transfinite_4.png"
plot_final_mesh(mesh, output_file=path)


meta_mesh.nodes_T4 = mesh.quad_coordinates
meta_mesh.faces_T4 = mesh.quad_faces


transfiniteInterpolator = QuadMeshGenerator(
    blocked_mesh, transfinite_divisions=5)
quad_mesh = transfiniteInterpolator.transfinite_mesh
mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

path = f"./figures/faces/mesh_transfinite_5.png"
plot_final_mesh(mesh, output_file=path)

meta_mesh.nodes_T5 = mesh.quad_coordinates
meta_mesh.faces_T5 = mesh.quad_faces


path = f"./figures/faces/mesh_streamlines.png"
plot_post_processed_streamline(mesh.streamlines, output_file=path)

torch.save(meta_mesh, 'meta_mesh.pt')
help(torch.save)
