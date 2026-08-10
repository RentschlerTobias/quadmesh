
from torch_geometric.data import Data
import torch
from tools.plotting_tools import *
import multiprocessing as mp
from data_generator import get_mesh
import matplotlib.pyplot as plt
import numpy as np
# Generate a new mesh

def cut_streamlines(Streamlines,Singularities):


    for key in Streamlines.keys():
    
        singularity_end = Streamlines[key]["s_in"]
    
        if singularity_end is not None:
            print(singularity_end)
            singularity_start = Streamlines[key]["s_out"]
            streamlines_staring =Singularities[singularity_end]["s_out"]
            best_angle = np.inf
    
            angle_in = Streamlines[key]["angle_in"]
            best_streamline= None
            for streamline in streamlines_staring:
                    angle_out= Streamlines[streamline]["angle_out"]
                    angle_diff =np.absolute(angle_out-angle_in) 
                    if angle_diff < best_angle:
                        best_angle=angle_diff
                        print(best_angle)
                        best_streamline = streamline
                    
            if best_angle < 2/3*np.pi:
                Streamlines[best_streamline]["cut_at_sing"] = singularity_start
                print(f'merge streamline {best_streamline} with sing {singularity_start}')
        
    for key in Streamlines.keys():
        id = Streamlines[key]["cut_at_sing"]
        if id is not None:
            sing_coord =  Singularity[id]['coords']
            if sing_coord is not None:
                streamline_coords = Streamlines[key]["coords"]
                distance = np.linalg.norm(streamline_coords - sing_coord, axis=1)
                distance_min_idx = np.argmin(distance)
                distance_min = distance[distance_min_idx]
                cutted_streamline = streamline[:distance_min_idx, :]
                new_streamline = np.vstack([cutted_streamline, termination_node])
                Streamlines[key]["coords"]= new_streamline
                Streamlines[key]["s_in"]=id 
    

    cutted_streamlines = []

    for key in Streamlines.keys():
        cutted_streamlines.append(Streamlines[key]["coords"])
    return cutted_streamlines

def streamline_post_processing(mesh):

    tol = 10e-3
    tol_2 = 10e-3
    streamlines = mesh.streamlines
    
    Singularity = {}
    Streamlines ={}

    mask_c0_nodes = mesh.tri_coordinates[:, 2] == 0
    c0_nodes = mesh.tri_coordinates[mask_c0_nodes, 0:2]
    streamline_termination_nodes = np.array([mesh.singularities_coords[sing] for sing in mesh.singularities_coords]+list(c0_nodes))


    for j in range(streamline_termination_nodes.shape[0]):
        Singularity[j] = {"s_in": [],"s_out":[],"coords": None}

    for i in range(len(streamlines)):
        Streamlines[i] = {"s_in": None,"s_out":None,"angle_in":None,"angle_out":None,"coords": None,"cut_at_sing": None}

    for i in range(len(streamlines)):
        streamline= streamlines[i]
        start = streamline[0]
        is_streamline_cutted = False
        for j in range(streamline_termination_nodes.shape[0]):
            termination_node= streamline_termination_nodes[j,:]
            distance_singularity = np.linalg.norm(start - termination_node)
            # Check if singularity is start point of streamline
            if distance_singularity < tol:

                dx = streamline[1, 0] - streamline[0, 0]
                dy = streamline[1, 1] - streamline[0, 1]

                Singularity[j]["s_out"].append(i)
                Singularity[j]["coords"]= termination_node

                Streamlines[i]["s_out"]=j
                Streamlines[i]["angle_out"]=np.arctan2(dy, dx)

                print('streamline starts at singularity')
                break
            else:
                distance = np.linalg.norm(streamline - termination_node, axis=1)
                distance_min_idx = np.argmin(distance)
                distance_min = distance[distance_min_idx]

                if distance_min < tol_2:

                    cutted_streamline = streamline[:distance_min_idx, :]
                    new_streamline = np.vstack([cutted_streamline, termination_node])

                    dx = streamline[distance_min_idx, 0] - streamline[distance_min_idx - 1, 0]
                    dy = streamline[distance_min_idx, 1] - streamline[distance_min_idx - 1, 1]
                    
                    Singularity[j]["s_in"].append(i)
                    Streamlines[i]["s_in"]= j
                    Streamlines[i]["angle_in"]=np.arctan2(dy, dx)
                    Streamlines[i]["coords"]=new_streamline

                    is_streamline_cutted = True

                    print('streamline cutted')
                    break
        if is_streamline_cutted == False:
            Streamlines[i]["coords"]=streamline

    return Streamlines, Singularity



## Identify and the streamline that missed a Sigulartiy during streamline integration
# def cut_missed_streamlines(streamlines_dicc): 
    
