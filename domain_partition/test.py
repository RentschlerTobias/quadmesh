
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools import Transfinite_Interpolation
from tools import MeshCheck, QuadPartitionValidator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import multiprocessing as mp
import time
def extract_mesh_data(tri_mesh,quad_mesh,block_mesh):

    # Extract features of the triangulated mesh incl. frame_field & streamline generation

    tri_coordinates     = tri_mesh.x
    tri_faces           = tri_mesh.faces
    tri_edges           = tri_mesh.edge_index
    tri_edges_attr      = tri_mesh.edge_attr
    tri_mesh_face_attr  = tri_mesh.face_attr
    streamlines         = tri_mesh.streamlines
    frame_field_angle   = tri_mesh.frame_field_iteration_number
    frame_field_u       = tri_mesh.u
    frame_field_iteration_number   = tri_mesh.frame_field_iteration_number

    frame_field_time = tri_mesh.time_frame_field_generator
    singularities = tri_mesh.singularities
    streamline_intersections_points= tri_mesh.streamline_intersections_points
    
    # Extract the block structure
    blocking_nodes = block_mesh.x
    blocking_faces = block_mesh.faces

    # Extract transfinite Interpolated mesh
    quad_coordinates    = quad_mesh.x
    quad_faces          = quad_mesh.faces
    quad_edges          = quad_mesh.edge_index
    
    final_mesh          = Data(blocking_nodes=blocking_nodes, blocking_faces = blocking_faces, quad_coordinates= quad_coordinates, quad_faces= quad_faces,quad_edges=quad_edges,streamline_intersections_points=streamline_intersections_points,singularities=singularities,frame_field_time=frame_field_time,frame_field_iteration_number=frame_field_iteration_number,frame_field_u=frame_field_u,frame_field_angle=frame_field_angle,streamlines=streamlines,tri_edges_attr=tri_edges_attr,tri_mesh_face_attr=tri_mesh_face_attr,tri_edges=tri_edges,tri_faces=tri_faces,tri_coordinates=tri_coordinates)

    return final_mesh


def get_mesh():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())
    
    try:
        print('started function NACA_airfoil()')
        airfoil                     = NACA_airfoil()
        random_lc                   =  0.04+ 0.02*np.random.rand()
        print('called function MeshGenerator')
        mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)

        print('called function FrameField')
        frameField                  = FrameField(mesh_gen.mesh)

        print('called function StreamlineGenerator')
        streamline                  = StreamlineGenerator(frameField.mesh)

        print('called function StreamlineSimplificator')
        streamlines_post_processed  = StreamlineSimplificator(streamline.mesh)

        print('called function streamlines_post_processed')
        blocked_mesh                = streamlines_post_processed.quad_mesh
        tri_mesh                    = streamline.mesh

        # --- Pre-Filter: Quad Partition Validator (Phase 1: strict=False) ---
        validator                   = QuadPartitionValidator(blocked_mesh, tri_mesh, strict=False)
        if not validator.is_valid():
            print('failed: blocked mesh invalid (pre-filter)')
            print('\n'.join(validator.diagnostics()))
            return None
        qs = validator.quality_score()
        print(f"Pre-filter quality: SJ_min={qs.get('scaled_jacobian_min', -1):.3f}, "
              f"angle=[{qs.get('min_interior_angle', -1):.1f}, {qs.get('max_interior_angle', -1):.1f}], "
              f"aspect={qs.get('edge_length_ratio_max', -1):.2f}")
        # --------------------------------------------------------------------

        print('called function Transfinite_Interpolation')
        transfiniteInterpolation    = Transfinite_Interpolation(blocked_mesh)

        quad_mesh                   = transfiniteInterpolation.quad_mesh
        mesh_check                  = MeshCheck(tri_mesh, quad_mesh, tol=0.015)
        is_valid                    = mesh_check.is_valid
        print(f'!!! \n area difference: \n {mesh_check.quad_area-mesh_check.tri_area}\n !!!')
        if is_valid == True:
            mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh) 
            print('succssess')
            return mesh
        else:
            print('failed')
            return None
    except Exception as e:
        print(f'\n domain partition failed: {e} \n')
 
