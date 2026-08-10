from gmsh import merge
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.interpolate import splprep, splev


class StreamlineMerging:

    def __init__(self, mesh: Data, verbose: bool = True):

        self.verbose = verbose
        self.Streamlines,self.Singularities = self.prepare_streamlines(mesh)
        self.new_streamlines                = self.merge_streamlines(self.Streamlines)

    def prepare_streamlines(self, mesh: Data):

        if self.verbose == True:
            print("\n function pre_processing \n")

        tol = 10e-3
        Singularities = {}
        Streamlines = {}

        mask_c0_nodes               = mesh.x[:, 2] == 0
        c0_nodes                    = mesh.x[mask_c0_nodes, 0:2]
        singularity_coords          = torch.tensor([mesh.singularities_coords[sing] for sing in mesh.singularities_coords], dtype=mesh.x.dtype)
        streamline_termination_nodes = torch.cat((c0_nodes, singularity_coords), 0)

        streamlines = mesh.streamlines

        for j in range(streamline_termination_nodes.size(0)):
            Singularities[j] = {"streamline_in": [], "streamline_out": [], "coords": streamline_termination_nodes[j], "is_boundary": j >= len(singularity_coords)}

        for i in range(len(streamlines)):
            streamline = torch.from_numpy(streamlines[i])
            Streamlines[i] = {"singularity_in": None, "singularity_out": None, "coords": streamline, "is_boundary": False,"angle_in":None, "angle_out": None }

        for i in range(len(streamlines)):
            streamline = torch.from_numpy(streamlines[i])
            start = streamline[0]
            end = streamline[-1]
            
            potential_cut = [] # store the indices where a streamline has a short distance to a termination node, cut it at the first index
            
            start_is_boundary = False
            end_is_boundary = False

            for j in range(streamline_termination_nodes.size(0)):

                termination_node    = streamline_termination_nodes[j, :]
                distance_start      = torch.linalg.norm(start - termination_node)
                distance_end        = torch.linalg.norm(end - termination_node)


                if distance_start < tol:

                    start_is_boundary = Singularities[j]["is_boundary"]
                    Singularities[j]["streamline_out"].append(i)
                    Streamlines[i]["singularity_out"] = j
                    
                    dx = streamline[1,0] - streamline[0,0]
                    dy = streamline[1,1] - streamline[0,1]
                    
                    Streamlines[i]["angle_out"] = torch.atan2(dy,dx)

                elif distance_end < tol:

                    end_is_boundary = Singularities[j]["is_boundary"]
                    Singularities[j]["streamline_in"].append(i)
                    Streamlines[i]["singularity_in"] = j
                    
                    dx = streamline[-1,0] - streamline[-2,0]
                    dy = streamline[-1,1] - streamline[-2,1]
                    
                    Streamlines[i]["angle_in"] = torch.atan2(dy,dx)
            if start_is_boundary and end_is_boundary:
                Streamlines[i]["is_boundary"] = True

        return Streamlines,Singularities

    def find_missed_streamline_endpoints(self, Streamlines, Singularities):
    
        for key in Singularities.keys():
            s_in = Singularities[key]["streamline_in"]

            if not s_in:
                continue

            s_out = Singularities[key]["streamline_out"]
            if not s_out:
                continue

            # Für jede eingehende Stromlinie
            for streamline_in in s_in:
                angle_in = Streamlines[streamline_in]['angle_in']

                # Finde beste passende ausgehende Stromlinie
                best_angle_diff = torch.pi  # Maximaler Winkelunterschied
                best_match_id = None

                for streamline_out in s_out:
                    angle_out = Streamlines[streamline_out]['angle_out']

                    # Normalisiere Winkeldifferenz auf [-π, π]
                    angle_diff = angle_in - angle_out
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                    angle_diff_abs = torch.abs(angle_diff)

                    # Prüfe ob dies ein Gegenstück ist (ungefähr 180° Unterschied)
                    if torch.abs(angle_diff_abs - torch.pi) < best_angle_diff:
                        best_angle_diff = torch.abs(angle_diff_abs - torch.pi)
                        best_match_id = streamline_out

                        if self.verbose:
                            print(f'Found potential match: streamline_in={streamline_in}, '
                                  f'streamline_out={best_match_id}, angle_diff={angle_diff_abs:.3f}')

                # Wenn eine passende Stromlinie gefunden wurde
                if best_match_id is not None and best_angle_diff < torch.pi/4:  # Toleranz von 45°
                    # Die ausgehende Stromlinie sollte an der anderen Singularität enden
                    target_sing_id = Streamlines[streamline_in]["singularity_out"]

                    # Überprüfe ob diese Stromlinie noch kein Ende hat
                    if Streamlines[best_match_id]["singularity_in"] is None and target_sing_id is not None:
                        target_singularity = Singularities[target_sing_id]['coords']
                        streamline_coords = Streamlines[best_match_id]["coords"]

                        # Berechne Distanz aller Punkte zur Ziel-Singularität
                        distances = torch.linalg.norm(
                            streamline_coords - target_singularity.unsqueeze(0), 
                            dim=1
                        )

                        # Finde nächsten Punkt
                        min_idx = torch.argmin(distances)
                        min_distance = distances[min_idx]

                        if self.verbose:
                            print(f'Streamline {best_match_id} passes singularity {target_sing_id} '
                                  f'at distance {min_distance:.6f} at index {min_idx}')

                        # Schneide nur wenn Distanz klein genug ist
                        if min_distance < 0.1:  # Toleranz anpassen
                            # Schneide Stromlinie am nächsten Punkt
                            cut_streamline = streamline_coords[:min_idx+1, :]

                            # Update Stromlinie
                            Streamlines[best_match_id]['coords'] = cut_streamline
                            Streamlines[best_match_id]["singularity_in"] = target_sing_id

                            # Update Singularität
                            if best_match_id not in Singularities[target_sing_id]["streamline_in"]:
                                Singularities[target_sing_id]["streamline_in"].append(best_match_id)

                            # Berechne neuen Eingangswinkel
                            if len(cut_streamline) > 1:
                                dx = cut_streamline[-1, 0] - cut_streamline[-2, 0]
                                dy = cut_streamline[-1, 1] - cut_streamline[-2, 1]
                                Streamlines[best_match_id]["angle_in"] = torch.atan2(dy, dx)

                            if self.verbose:
                                print(f'Connected streamline {best_match_id} to singularity {target_sing_id}')

        return Streamlines, Singularities
                
    def merge_streamlines(self, Streamlines: dict):

        if self.verbose:
            print(f"\n started function merge_streamlines\n")

        merge_pairs = self.search_duplicated_streamlines(Streamlines)

        if self.verbose:
            print(f"\n found {len(merge_pairs)} streamlines to merge\n")

        merged_keys = set()
        for pair in merge_pairs:
            merged_keys.add(pair[0])
            merged_keys.add(pair[1])
        
        new_streamlines = []

        for key in Streamlines.keys():
            if key not in merged_keys:
                new_streamlines.append(np.array(Streamlines[key]["coords"]))

        for pair in merge_pairs:
            streamline_1 = Streamlines[pair[0]]["coords"]
            streamline_2 = Streamlines[pair[1]]["coords"]

            merged_streamline = self.interpolate_streamlines(streamline_1, streamline_2)
            new_streamlines.append(merged_streamline)

        return new_streamlines

    def search_duplicated_streamlines(self, Streamlines: dict):

        if self.verbose == True:
            print(f"\n started function search_duplicated_streamlines\n")

        merge_pairs = []

        for key_i in Streamlines.keys():
            start_i = Streamlines[key_i]["singularity_out"]
            end_i   = Streamlines[key_i]["singularity_in"]

            if end_i is None:
                continue

            for key_j in Streamlines.keys():
                start_j = Streamlines[key_j]["singularity_out"]
                end_j   = Streamlines[key_j]["singularity_in"]

                if end_j is None:
                    continue

                if Streamlines[key_i]["is_boundary"] and Streamlines[key_j]["is_boundary"]:
                    continue

                if start_i == end_j and start_j == end_i:

                    if [key_i, key_j] not in merge_pairs and [key_j, key_i] not in merge_pairs:
                        merge_pairs.append([key_i, key_j])

        return merge_pairs

    def get_streamlines_as_splines(self, streamlines):
        splines = []

        for streamline in streamlines:
            streamline = np.array(streamline)
            x = streamline[:, 0]
            y = streamline[:, 1]
            m = x.shape[0]

            # choose degree: min(3, m-1)
            k = min(3, m-1)

            # At least 2 points are needed
            if m < 2:
                continue  # or raise an error

            tck, u = splprep([x, y], s=0, k=k)
            splines.append([tck, u])

        return splines

    def interpolate_streamlines(self, streamline_ij, streamline_ji, num_points=100):
        # Convert streamlines to splines
        splines_ij = self.get_streamlines_as_splines([streamline_ij])
        splines_ji = self.get_streamlines_as_splines([streamline_ji])

        # Extract splines (assuming get_streamlines_as_splines returns [tck, u] for each streamline)
        tck_ij, u_ij = splines_ij[0]
        tck_ji, u_ji = splines_ji[0]

        # Generate uniform parameter values
        u_new = np.linspace(0, 1, num_points)

        # Evaluate splines at new parameter values
        x_ij, y_ij = splev(u_new, tck_ij)
        x_ji, y_ji = splev(u_new, tck_ji)

        x_merged = (1 - u_new) * x_ij + u_new * x_ji
        y_merged = (1 - u_new) * y_ij + u_new * y_ji
        # Stack and return merged streamline
        merged_streamline = np.vstack((x_merged, y_merged)).T

        return merged_streamline
