
from torch_geometric.data import Data
import torch
from tools.plotting_tools import *
import multiprocessing as mp
from data_generator import get_mesh
import matplotlib.pyplot as plt
import numpy as np
# Generate a new mesh


def get_new_mesh():
    is_valid = False

    while is_valid == False:
        block_mesh = mp.Pool(1).apply_async(get_mesh).get(timeout=300)

        if block_mesh is not None:
            is_valid = True

    file = f"./figures/streamlines/quad_mesh_new.png"
    plot_final_mesh(block_mesh, output_file=file)

    file_streamlines = f"./figures/streamlines/streamlines_new_mesh.png"
    plot_streamlines(block_mesh, output_file=file_streamlines)

    return block_mesh


def load_mesh():
    extension = 'post_processing'
    path = f'./saved_meshes/checkpoints_test/mesh_{extension}.pt'

    block_mesh = torch.load(path, weights_only=False)
    return block_mesh


def cut_streamlines(Singularities, Streamlines):
    for key in Streamlines.keys():
        singularity_end = Streamlines[key]["s_in"]

        if singularity_end is not None:
            print(singularity_end)
            singularity_start = Streamlines[key]["s_out"]
            streamlines_starting = Singularities[singularity_end]["s_out"]
            start_sing_coords = Singularities[singularity_start]["coords"] if singularity_start is not None else None
            end_sing_coords = Singularities[singularity_end]["coords"]

            if start_sing_coords is not None:
                best_angle_diff = np.inf
                best_streamline = None

                dx = start_sing_coords[0] - end_sing_coords[0]
                dy = start_sing_coords[1] - end_sing_coords[1]
                angle_in = np.arctan2(dy, dx)

                for streamline_id in streamlines_starting:
                    if streamline_id in Streamlines:
                        angle_out = Streamlines[streamline_id]["angle_out"]

                        angle_diff = np.abs(angle_out - angle_in)
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                        if angle_diff < best_angle_diff:
                            best_angle_diff = angle_diff
                            best_streamline = streamline_id
                            print(best_angle_diff)

                if best_angle_diff < (2 / 3) * np.pi and best_streamline is not None:
                    Streamlines[best_streamline]["cut_at_sing"] = singularity_start
                    print(f'merge streamline {
                          best_streamline} with sing {singularity_start}')

    for key in Streamlines.keys():
        if "cut_at_sing" in Streamlines[key] and Streamlines[key]["cut_at_sing"] is not None:
            cut_singularity_id = Streamlines[key]["cut_at_sing"]

            if cut_singularity_id in Singularities:
                sing_coords = Singularities[cut_singularity_id]["coords"]
                streamline_coords = Streamlines[key]["coords"]

                if streamline_coords is not None and len(streamline_coords) > 0:
                    distances = np.linalg.norm(
                        streamline_coords - sing_coords.reshape(1, -1), axis=1)
                    cut_index = np.argmin(distances)

                    if cut_index > 0:
                        cut_streamline = streamline_coords[:cut_index + 1]
                        cut_streamline[-1] = sing_coords

                        Streamlines[key]["coords"] = cut_streamline
                        Streamlines[key]["s_in"] = cut_singularity_id

    return Streamlines


def streamline_post_processing(mesh):

    tol = 10e-3
    tol_2 = 10e-3
    streamlines = mesh.streamlines

    Singularity = {}
    Streamlines = {}

    mask_c0_nodes = mesh.tri_coordinates[:, 2] == 0
    c0_nodes = mesh.tri_coordinates[mask_c0_nodes, 0:2]
    streamline_termination_nodes = np.array(
        [mesh.singularities_coords[sing] for sing in mesh.singularities_coords] + list(c0_nodes))

    for j in range(streamline_termination_nodes.shape[0]):
        Singularity[j] = {"s_in": [], "s_out": [], "coords": None}

    for i in range(len(streamlines)):
        Streamlines[i] = {"s_in": None, "s_out": None, "angle_in": None,
                          "angle_out": None, "coords": None, "cut_at_sing": None}

    for i in range(len(streamlines)):
        streamline = streamlines[i]
        start = streamline[0]
        is_streamline_cutted = False
        for j in range(streamline_termination_nodes.shape[0]):
            termination_node = streamline_termination_nodes[j, :]
            distance_singularity = np.linalg.norm(start - termination_node)
            # Check if singularity is start point of streamline
            if distance_singularity < tol:

                dx = streamline[1, 0] - streamline[0, 0]
                dy = streamline[1, 1] - streamline[0, 1]

                Singularity[j]["s_out"].append(i)
                Singularity[j]["coords"] = termination_node

                Streamlines[i]["s_out"] = j
                Streamlines[i]["angle_out"] = np.arctan2(dy, dx)

                print('streamline starts at singularity')
                break
            else:
                distance = np.linalg.norm(
                    streamline - termination_node, axis=1)
                distance_min_idx = np.argmin(distance)
                distance_min = distance[distance_min_idx]

                if distance_min < tol_2:

                    cutted_streamline = streamline[:distance_min_idx, :]
                    new_streamline = np.vstack(
                        [cutted_streamline, termination_node])

                    dx = streamline[distance_min_idx, 0] - \
                        streamline[distance_min_idx - 1, 0]
                    dy = streamline[distance_min_idx, 1] - \
                        streamline[distance_min_idx - 1, 1]

                    Singularity[j]["s_in"].append(i)
                    Streamlines[i]["s_in"] = j
                    Streamlines[i]["angle_in"] = np.arctan2(dy, dx)
                    Streamlines[i]["coords"] = new_streamline

                    is_streamline_cutted = True

                    print('streamline cutted')
                    break
        if is_streamline_cutted == False:
            Streamlines[i]["coords"] = streamline

    return Streamlines, Singularity


# Identify and the streamline that missed a Sigulartiy during streamline integration
# def cut_missed_streamlines(streamlines_dicc):
