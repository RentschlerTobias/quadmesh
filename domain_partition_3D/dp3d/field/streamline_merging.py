from gmsh import merge
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.interpolate import splprep, splev


class StreamlineMerging:

    def __init__(self, mesh: Data, verbose: bool = True):

        self.verbose = verbose
        self.Streamlines,self.Singularities = self.prepare_streamlines(mesh)
        self.new_streamlines                = self.merge_streamlines(
            self.Streamlines, self.Singularities,
            getattr(mesh, "expected_separatrices", None), mesh)

    def prepare_streamlines(self, mesh: Data):

        if self.verbose == True:
            print("\n function pre_processing \n")

        tol = 10e-3
        Singularities = {}
        Streamlines = {}

        mask_c0_nodes               = mesh.x[:, 2] == 0
        c0_nodes                    = mesh.x[mask_c0_nodes, 0:2]
        def _to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.to(mesh.x.dtype)
            return torch.tensor(v, dtype=mesh.x.dtype)
        singularity_coords = torch.stack([_to_tensor(mesh.singularities_coords[sing]) for sing in mesh.singularities_coords])
        streamline_termination_nodes = torch.cat((c0_nodes, singularity_coords), 0)
        self.streamline_termination_nodes = streamline_termination_nodes

        streamlines = mesh.streamlines

        for j in range(streamline_termination_nodes.size(0)):
            Singularities[j] = {"streamline_in": [], "streamline_out": [], "coords": streamline_termination_nodes[j], "is_boundary": j < len(c0_nodes)}

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

                    if target_sing_id is None:
                        continue

                    target_singularity = Singularities[target_sing_id]['coords']
                    streamline_coords = Streamlines[best_match_id]["coords"]

                    # Berechne Distanz aller Punkte zur Ziel-Singularität
                    distances = torch.linalg.norm(
                        streamline_coords - target_singularity.unsqueeze(0),
                        dim=1
                    )
                    min_idx = torch.argmin(distances)
                    min_distance = distances[min_idx]

                    # CASE 1: Streamline hat noch kein Ende -> direkt verbinden
                    if Streamlines[best_match_id]["singularity_in"] is None:
                        if min_distance < 0.1:
                            if self.verbose:
                                print(f'[Case 1] Streamline {best_match_id} connects to singularity {target_sing_id} '
                                      f'(distance {min_distance:.6f})')
                            self._cut_streamline_at_singularity(
                                Streamlines, Singularities, best_match_id, target_sing_id, min_idx
                            )
                        continue

                    # CASE 3 (Xiao paper): Streamline hat bereits ein Ende,
                    # aber Ziel-Singularität liegt "früher" auf dem Weg
                    current_sing_id = Streamlines[best_match_id]["singularity_in"]
                    current_singularity = Singularities[current_sing_id]['coords']
                    distances_current = torch.linalg.norm(
                        streamline_coords - current_singularity.unsqueeze(0),
                        dim=1
                    )
                    min_idx_current = torch.argmin(distances_current)

                    # "Früher" = Ziel-Singularität wird vor aktueller End-Singularität erreicht
                    if min_idx < min_idx_current and min_distance < 0.1:
                        if self.verbose:
                            print(f'[Case 3] Streamline {best_match_id} passes earlier singularity {target_sing_id} '
                                  f'(idx={min_idx}, dist={min_distance:.6f}) before current end {current_sing_id} '
                                  f'(idx={min_idx_current})')
                        # Altes Ende entfernen; der Rest hinter dem
                        # Schnittpunkt wird verworfen
                        if best_match_id in Singularities[current_sing_id]["streamline_in"]:
                            Singularities[current_sing_id]["streamline_in"].remove(best_match_id)
                        # Neues Ende setzen
                        self._cut_streamline_at_singularity(
                            Streamlines, Singularities, best_match_id, target_sing_id, min_idx
                        )

        return Streamlines, Singularities

    def _cut_streamline_at_singularity(self, Streamlines, Singularities, sl_id, sing_id, cut_idx):
        """Cut streamline at given index and connect to singularity."""
        coords = Streamlines[sl_id]["coords"]
        cut_streamline = coords[:cut_idx+1, :]

        Streamlines[sl_id]['coords'] = cut_streamline
        Streamlines[sl_id]["singularity_in"] = sing_id

        if sl_id not in Singularities[sing_id]["streamline_in"]:
            Singularities[sing_id]["streamline_in"].append(sl_id)

        if len(cut_streamline) > 1:
            dx = cut_streamline[-1, 0] - cut_streamline[-2, 0]
            dy = cut_streamline[-1, 1] - cut_streamline[-2, 1]
            Streamlines[sl_id]["angle_in"] = torch.atan2(dy, dx)

        if self.verbose:
            print(f'  -> Cut streamline {sl_id} at singularity {sing_id}')
                
    def merge_streamlines(self, Streamlines: dict, Singularities: dict = None,
                          expected_separatrices: dict = None, mesh: Data = None):
        """Stack-based streamline merging following Xiao et al. Algorithm 2."""

        if self.verbose:
            print(f"\n started function merge_streamlines (stack-based)\n")

        if Singularities is None:
            return self._merge_streamlines_legacy(Streamlines)

        # Deep-copy mutable structures to avoid side-effects during stack iteration
        SL = {k: {**v, "coords": v["coords"].clone()} for k, v in Streamlines.items()}
        SG = {k: {
            "streamline_in": list(v["streamline_in"]),
            "streamline_out": list(v["streamline_out"]),
            "coords": v["coords"].clone(),
            "is_boundary": v["is_boundary"]
        } for k, v in Singularities.items()}

        stack = list(SG.keys())
        processed_pairs = set()
        processed_endpoint_pairs = set()
        next_id = max(SL.keys()) + 1 if SL else 0
        max_iterations = 1000
        iteration = 0

        def _remove_streamline(sl_id):
            if sl_id not in SL:
                return
            sl = SL[sl_id]
            s_out = sl.get("singularity_out")
            s_in = sl.get("singularity_in")
            if s_out is not None and sl_id in SG[s_out]["streamline_out"]:
                SG[s_out]["streamline_out"].remove(sl_id)
            if s_in is not None and sl_id in SG[s_in]["streamline_in"]:
                SG[s_in]["streamline_in"].remove(sl_id)
            del SL[sl_id]

        while stack:
            iteration += 1
            if iteration > max_iterations:
                if self.verbose:
                    print(f"WARNING: Max iterations ({max_iterations}) reached, breaking to avoid infinite loop")
                break
            s_c = stack.pop(0)

            gamma_out = [sl_id for sl_id in SG[s_c]["streamline_out"]]
            gamma_in = [sl_id for sl_id in SG[s_c]["streamline_in"]]

            if not gamma_out or not gamma_in:
                continue

            for sl_j in list(gamma_in):
                if sl_j not in SL:
                    continue

                # Eq. 9: angle-based matching criterion
                best_match = None
                best_angle_diff = float('inf')

                angle_in_j = float(SL[sl_j]["angle_in"])

                for sl_i in gamma_out:
                    if sl_i not in SL:
                        continue
                    pair = tuple(sorted([sl_j, sl_i]))
                    if pair in processed_pairs:
                        continue

                    angle_out_i = float(SL[sl_i]["angle_out"])
                    angle_diff = abs(angle_in_j - angle_out_i)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                    # Same-arm pairs are anti-parallel at s_c (diff ~ pi);
                    # a parallel pair is the opposite arm, never a match.
                    valence = len(gamma_out)
                    if valence == 5:
                        valid = (4 * np.pi / 5 <= angle_diff <= 6 * np.pi / 5)
                    else:
                        valid = (2 * np.pi / 3 <= angle_diff <= 4 * np.pi / 3)

                    score = abs(angle_diff - np.pi)
                    if valid and score < best_angle_diff:
                        best_angle_diff = score
                        best_match = sl_i

                if best_match is None:
                    continue

                pair = tuple(sorted([sl_j, best_match]))
                processed_pairs.add(pair)

                # Fig. 8: determine topological case
                s_i = SL[best_match]["singularity_in"]
                s_j = SL[sl_j]["singularity_out"]

                if s_i is None or s_j is None:
                    continue

                if s_i == s_j:
                    # Fig. 8a Case 1: both streamlines connect s_c and s_i=s_j bidirectionally
                    if self.verbose:
                        print(f"  [Case 1] S{s_c}: merge sl {best_match} (S{s_c}->S{s_i}) + sl {sl_j} (S{s_j}->S{s_c})")

                    merged = self.interpolate_streamlines(
                        np.array(SL[best_match]["coords"]),
                        np.array(SL[sl_j]["coords"])
                    )

                    loop_dist = np.linalg.norm(merged[0] - merged[-1])
                    if loop_dist < 0.05:
                        if self.verbose:
                            print(f"    -> loop detected (dist={loop_dist:.4f}), keeping longer original")
                        if len(SL[best_match]["coords"]) >= len(SL[sl_j]["coords"]):
                            kept_id = best_match
                        else:
                            kept_id = sl_j
                        _remove_streamline(best_match if kept_id != best_match else sl_j)
                        continue

                    # Register merged streamline
                    SL[next_id] = {
                        "coords": torch.tensor(merged),
                        "singularity_out": s_c,
                        "singularity_in": s_i,
                        "is_boundary": False,
                        "angle_out": SL[best_match]["angle_out"],
                        "angle_in": SL[sl_j]["angle_in"]
                    }
                    SG[s_c]["streamline_out"].append(next_id)
                    SG[s_i]["streamline_in"].append(next_id)
                    next_id += 1

                    _remove_streamline(best_match)
                    _remove_streamline(sl_j)

                else:
                    gamma_i_coords = SL[best_match]["coords"]
                    s_j_coords = SG[s_j]["coords"]

                    # Project s_j onto gamma_i: find closest point
                    distances = torch.linalg.norm(
                        gamma_i_coords - s_j_coords.unsqueeze(0), dim=1
                    )
                    min_idx = int(torch.argmin(distances))

                    # Case 2/3 only applies when s_j actually lies on
                    # gamma_i; splitting at a far projection fabricates arms.
                    if float(distances[min_idx]) >= 0.1:
                        continue

                    endpoint_pair = frozenset({s_c, s_j})
                    if endpoint_pair in processed_endpoint_pairs:
                        if self.verbose:
                            print(f"  [skip] endpoint pair {s_c}-{s_j} already processed")
                        continue
                    processed_endpoint_pairs.add(endpoint_pair)

                    if self.verbose:
                        print(f"  [Case 2/3] S{s_c}: sl {best_match} (S{s_c}->S{s_i}) + sl {sl_j} (S{s_j}->S{s_c})")

                    # Split gamma_i at the projection point; the part beyond
                    # it (projection -> s_i) is discarded
                    gamma_i1 = gamma_i_coords[:min_idx + 1, :].clone()   # s_c -> projection

                    # Merge gamma_i1 (which now effectively ends at s_j) with gamma_j
                    merged = self.interpolate_streamlines(
                        np.array(gamma_i1),
                        np.array(SL[sl_j]["coords"])
                    )

                    SL[next_id] = {
                        "coords": torch.tensor(merged),
                        "singularity_out": s_c,
                        "singularity_in": s_j,
                        "is_boundary": False,
                        "angle_out": SL[best_match]["angle_out"],
                        "angle_in": SL[sl_j]["angle_in"]
                    }
                    SG[s_c]["streamline_out"].append(next_id)
                    SG[s_j]["streamline_in"].append(next_id)
                    next_id += 1

                    _remove_streamline(best_match)
                    _remove_streamline(sl_j)

                    if s_j not in stack:
                        stack.append(s_j)
                        if self.verbose:
                            print(f"    -> pushed S{s_j} onto stack for reprocessing")

        if mesh is not None and expected_separatrices is not None:
            self._ensure_separatrix_count(SL, SG, expected_separatrices, mesh, Streamlines)

        new_streamlines = []
        for key in sorted(SL.keys()):
            coords = SL[key]["coords"]
            if isinstance(coords, torch.Tensor):
                coords = coords.numpy()
            new_streamlines.append(np.array(coords))

        if self.verbose:
            print(f"\n Streamline merging complete: {len(Streamlines)} -> {len(new_streamlines)} streamlines\n")

        return new_streamlines

    def _merge_streamlines_legacy(self, Streamlines: dict):
        merge_pairs = self.search_duplicated_streamlines(Streamlines)

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

            loop_distance = np.linalg.norm(merged_streamline[0] - merged_streamline[-1])
            if loop_distance < 0.05:
                if len(streamline_1) >= len(streamline_2):
                    new_streamlines.append(np.array(streamline_1))
                else:
                    new_streamlines.append(np.array(streamline_2))
            else:
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

                n_sing = len([s for s in Streamlines.values() if s['singularity_out'] is not None])
                both_real_boundary = (
                    Streamlines[key_i]["is_boundary"] and start_i >= n_sing and end_i >= n_sing and
                    Streamlines[key_j]["is_boundary"] and start_j >= n_sing and end_j >= n_sing
                )
                if both_real_boundary:
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
        if len(streamline_ij) < 2 or len(streamline_ji) < 2:
            if len(streamline_ij) >= len(streamline_ji):
                return np.array(streamline_ij)
            return np.array(streamline_ji)

        a_start = np.asarray(streamline_ij[0], float)
        a_end = np.asarray(streamline_ij[-1], float)
        b_start = np.asarray(streamline_ji[0], float)
        b_end = np.asarray(streamline_ji[-1], float)

        same_path_reversed = (
            np.linalg.norm(a_start - b_end) < 0.05 and
            np.linalg.norm(a_end - b_start) < 0.05
        )
        if same_path_reversed:
            streamline_ji = streamline_ji[::-1]

        splines_ij = self.get_streamlines_as_splines([streamline_ij])
        splines_ji = self.get_streamlines_as_splines([streamline_ji])

        if not splines_ij or not splines_ji:
            if len(streamline_ij) >= len(streamline_ji):
                return np.array(streamline_ij)
            return np.array(streamline_ji)

        tck_ij, u_ij = splines_ij[0]
        tck_ji, u_ji = splines_ji[0]

        u_new = np.linspace(0, 1, num_points)

        x_ij, y_ij = splev(u_new, tck_ij)
        x_ji, y_ji = splev(u_new, tck_ji)

        x_merged = (1 - u_new) * x_ij + u_new * x_ji
        y_merged = (1 - u_new) * y_ij + u_new * y_ji
        merged_streamline = np.vstack((x_merged, y_merged)).T

        return merged_streamline

    def _ensure_separatrix_count(self, SL, SG, expected_separatrices, mesh, original_Streamlines):
        """Re-check separatrix count after merge and insert missing separatrices."""
        if not expected_separatrices:
            return

        face_map = {}
        for face_id, coord in mesh.singularities_coords.items():
            coord_t = torch.tensor(coord, dtype=torch.float)
            for s_idx, s_data in SG.items():
                if s_data["is_boundary"]:
                    continue
                if torch.linalg.norm(s_data["coords"] - coord_t) < 1e-4:
                    face_map[s_idx] = face_id
                    break

        for s_idx, s_data in SG.items():
            if s_data["is_boundary"]:
                continue
            face_id = face_map.get(s_idx)
            if face_id is None:
                continue
            expected = expected_separatrices.get(face_id)
            if expected is None:
                continue

            # Count separatrices by distinct direction, not by length (Kowalski
            # et al.): a genuine separatrix that terminates early on a nearby
            # singularity is a short polyline but a real arm, so a length cutoff
            # would drop it and fabricate a spurious "missing". Arms sharing a
            # direction (a stub duplicating a longer arm, or a split fragment)
            # collapse to one.
            all_ids = s_data["streamline_in"] + s_data["streamline_out"]
            distinct = self._distinct_direction_arms(SL, s_idx, all_ids)
            actual = len(distinct)
            missing = expected - actual
            if missing <= 0:
                continue

            if self.verbose:
                _c = SG[s_idx]["coords"]
                print(f"  [Post-merge check] S{s_idx} at "
                      f"({float(_c[0]):.3f}, {float(_c[1]):.3f}): "
                      f"expected={expected}, actual={actual} "
                      f"(distinct directions), missing={missing}")

            current_targets = self._singularity_targets(SL, s_idx, distinct)
            angles = self._singularity_streamline_angles(SL, s_idx, distinct)
            inserted = 0
            attempts = 0
            while inserted < missing and attempts < 5:
                attempts += 1
                if len(angles) < 2:
                    gap_mid = 0.0
                else:
                    sorted_angles = sorted(angles)
                    gaps = []
                    for i in range(len(sorted_angles)):
                        a1 = sorted_angles[i]
                        a2 = sorted_angles[(i + 1) % len(sorted_angles)]
                        gap = (a2 - a1) % (2 * np.pi)
                        gaps.append((gap, i))
                    gaps.sort(key=lambda x: x[0], reverse=True)
                    gap_mid = None
                    for biggest_gap, idx in gaps:
                        a1 = sorted_angles[idx]
                        a2 = sorted_angles[(idx + 1) % len(sorted_angles)]
                        candidate = (a1 + biggest_gap / 2) % (2 * np.pi)
                        if all(abs(candidate - angle) > 0.1 for angle in angles):
                            gap_mid = candidate
                            break
                    if gap_mid is None:
                        gap_mid = (a1 + biggest_gap / 2) % (2 * np.pi)

                best_candidate = self._best_candidate_for_gap(
                    SL, SG, s_idx, gap_mid, current_targets, original_Streamlines, mesh)
                if best_candidate is None:
                    angles.append(gap_mid)
                    continue

                new_sl, target, new_angle = best_candidate
                new_id = max(SL.keys()) + 1 if SL else 0
                SL[new_id] = {
                    "coords": torch.tensor(new_sl, dtype=torch.float),
                    "singularity_out": s_idx,
                    "singularity_in": target if target is not None else None,
                    "is_boundary": False,
                    "angle_out": torch.tensor(new_angle, dtype=torch.float),
                    "angle_in": None,
                }
                if target is not None:
                    SG[target]["streamline_in"].append(new_id)
                SG[s_idx]["streamline_out"].append(new_id)
                angles.append(new_angle)
                current_targets.add(target)
                inserted += 1
                if self.verbose:
                    _t = SG[target]["coords"] if target is not None else None
                    _ts = (f"({float(_t[0]):.3f}, {float(_t[1]):.3f})"
                           if _t is not None else "None")
                    _e = new_sl[-1]
                    print(f"    -> inserted new streamline {new_id} from S{s_idx} "
                          f"to target S{target} {_ts}, gap_mid={gap_mid:.2f} rad, "
                          f"end=({float(_e[0]):.3f}, {float(_e[1]):.3f}), "
                          f"len={len(new_sl)}")

    def _singularity_targets(self, SL, s_idx, sl_ids):
        """Return the set of target singularity indices for streamlines at s_idx."""
        targets = set()
        for sl_id in sl_ids:
            sl_data = SL[sl_id]
            if sl_data["singularity_out"] == s_idx:
                targets.add(sl_data["singularity_in"])
            elif sl_data["singularity_in"] == s_idx:
                targets.add(sl_data["singularity_out"])
        return targets

    def _best_candidate_for_gap(self, SL, SG, s_idx, gap_mid, current_targets, original_Streamlines, mesh):
        """Try to reuse an original streamline or integrate a new one in the given gap."""
        direction = torch.tensor([np.cos(gap_mid), np.sin(gap_mid)], dtype=torch.float)
        singularity_coords = SG[s_idx]["coords"]

        best_orig = None
        best_orig_angle_diff = float('inf')
        for sl_id, sl_data in original_Streamlines.items():
            if sl_data["singularity_out"] == s_idx:
                target = sl_data["singularity_in"]
                other_end = sl_data["coords"][-1]
            elif sl_data["singularity_in"] == s_idx:
                target = sl_data["singularity_out"]
                other_end = sl_data["coords"][0]
            else:
                continue
            if target == s_idx or target in current_targets:
                continue
            if len(sl_data["coords"]) < 2:
                continue
            vec = other_end - singularity_coords
            angle = float(np.arctan2(float(vec[1]), float(vec[0])))
            angle_diff = abs((angle - gap_mid + np.pi) % (2 * np.pi) - np.pi)
            if angle_diff < best_orig_angle_diff:
                best_orig_angle_diff = angle_diff
                best_orig = (sl_id, sl_data, target)

        if best_orig is not None and best_orig_angle_diff < np.pi / 4:
            _, sl_data, target = best_orig
            coords = np.array(sl_data["coords"])
            if sl_data["singularity_out"] != s_idx:
                coords = coords[::-1].copy()
            return coords, target, float(np.arctan2(coords[1, 1] - coords[0, 1],
                                                    coords[1, 0] - coords[0, 0]))

        # Pick the initial arm in a well-defined field region: the cross field is
        # degenerate right at the singularity, so a 0.001 probe can seed the wrong
        # branch. Sample a bit further out along the gap and take the cross arm
        # aligned with it, then integrate from close to the singularity so the arm
        # still attaches there.
        best_dir = None
        for off in (0.02, 0.012, 0.006, 0.001):
            sample = singularity_coords + off * direction
            f = self._find_containing_face(sample, mesh)
            if f is not None:
                best_dir, _ = self._get_best_cross_vector(sample, direction, mesh, f)
                break
        if best_dir is None:
            return None
        probe = singularity_coords + 0.001 * direction
        if self._find_containing_face(probe, mesh) is None:
            return None
        new_sl = self._integrate_streamline(probe, best_dir, mesh)

        # Truncate at the first singularity the arm passes: a separatrix
        # terminates there, it must not integrate through onto a later one.
        arr = np.asarray(new_sl)
        cut = None
        for t_idx, t_data in SG.items():
            if t_idx == s_idx:
                continue
            d = np.linalg.norm(arr - t_data["coords"].numpy(), axis=1)
            hits = np.nonzero(d < 0.05)[0]
            if len(hits) and (cut is None or hits[0] < cut):
                cut = int(hits[0])
        if cut is not None:
            new_sl = arr[:cut + 1]
        if len(new_sl) < 10:
            return None

        # Accept the nearest singularity to the endpoint -- including a boundary
        # singularity: a separatrix legitimately terminates on the boundary.
        end = new_sl[-1]
        target = None
        min_dist = float('inf')
        for t_idx, t_data in SG.items():
            dist = float(np.linalg.norm(t_data["coords"].numpy() - end))
            if dist < min_dist and dist < 0.05:
                min_dist = dist
                target = t_idx
        if target is None or target == s_idx or target in current_targets:
            return None

        new_angle = float(torch.atan2(best_dir[1], best_dir[0]))
        return new_sl, target, new_angle

    def _distinct_direction_arms(self, SL, s_idx, sl_ids, tol_deg=15.0):
        """Representative streamline ids, one per distinct separatrix direction
        at s_idx. Arms whose tangent at the singularity differ by < tol_deg are
        the same separatrix; the longest is kept so a genuine short arm survives
        while a duplicating stub or split fragment collapses onto it."""
        tol = np.radians(tol_deg)
        arms = []
        for sl_id in sl_ids:
            sl_data = SL[sl_id]
            coords = sl_data["coords"]
            if len(coords) < 2:
                continue
            # arm direction pointing away from s_idx for both orientations,
            # so opposite rays stay distinct and same rays collapse
            if sl_data["singularity_out"] == s_idx:
                dx = coords[1, 0] - coords[0, 0]
                dy = coords[1, 1] - coords[0, 1]
            elif sl_data["singularity_in"] == s_idx:
                dx = coords[-2, 0] - coords[-1, 0]
                dy = coords[-2, 1] - coords[-1, 1]
            else:
                continue
            arms.append((len(coords), float(torch.atan2(dy, dx)), sl_id))

        arms.sort(reverse=True)   # longest first -> cluster representative
        reps = []
        for _len, ang, sl_id in arms:
            if any(abs((ang - r_ang + np.pi) % (2 * np.pi) - np.pi) < tol
                   for r_ang, _ in reps):
                continue
            reps.append((ang, sl_id))
        return [sl_id for _, sl_id in reps]

    def _singularity_streamline_angles(self, SL, s_idx, sl_ids):
        """Collect angles of the given streamlines at the singularity end touching s_idx."""
        angles = []
        for sl_id in sl_ids:
            sl_data = SL[sl_id]
            coords = sl_data["coords"]
            if len(coords) < 2:
                continue
            # outward-pointing direction at s_idx for both orientations
            if sl_data["singularity_out"] == s_idx:
                dx = coords[1, 0] - coords[0, 0]
                dy = coords[1, 1] - coords[0, 1]
                angles.append(float(torch.atan2(dy, dx)))
            elif sl_data["singularity_in"] == s_idx:
                dx = coords[-2, 0] - coords[-1, 0]
                dy = coords[-2, 1] - coords[-1, 1]
                angles.append(float(torch.atan2(dy, dx)))
        return angles

    def _integrate_streamline(self, start_point, start_direction, mesh, step_size=None):
        """Integrate a single streamline from start_point in start_direction."""
        if step_size is None:
            step_size = 10e-3 / 3
        current_point = start_point.clone().detach()
        current_direction = start_direction / torch.norm(start_direction)
        current_face_idx = self._find_containing_face(current_point, mesh)
        if current_face_idx is None:
            return np.array([start_point.numpy()])

        streamline = [start_point.numpy()]
        max_steps = 1000
        for _ in range(max_steps):
            v_current, _ = self._get_best_cross_vector(current_point, current_direction, mesh, current_face_idx)
            if v_current is None:
                break
            v_current = v_current / torch.norm(v_current)

            predictor_point = current_point + step_size * v_current
            predicted_face_id = self._find_containing_face(predictor_point, mesh)
            terminate, end_point = self._check_termination_criteria(predictor_point, predicted_face_id, start_point)
            if terminate:
                streamline.append(end_point.numpy())
                break

            v_predictor, _ = self._get_best_cross_vector(predictor_point, v_current, mesh, predicted_face_id)
            if v_predictor is None:
                break
            v_predictor = v_predictor / torch.norm(v_predictor)

            average_direction = (v_current + v_predictor) / 2.0
            average_direction = average_direction / torch.norm(average_direction)
            next_point = current_point + step_size * average_direction
            new_face_id = self._find_containing_face(next_point, mesh)
            terminate, end_point = self._check_termination_criteria(next_point, new_face_id, start_point)
            if terminate:
                streamline.append(end_point.numpy())
                break

            current_point = next_point
            current_direction = average_direction
            current_face_idx = new_face_id
            streamline.append(current_point.numpy())

        return np.array(streamline)

    def _check_termination_criteria(self, point, face_idx, origin):
        if face_idx is None:
            return True, point
        distances = torch.linalg.norm(self.streamline_termination_nodes - point, dim=1)
        min_distance = torch.min(distances)
        if min_distance < 10e-3:
            idx = int(torch.argmin(distances))
            termination_node = self.streamline_termination_nodes[idx, :]
            if torch.linalg.norm(point - origin) < 10e-3:
                return False, None
            return True, termination_node
        return False, None

    def _find_containing_face(self, point, mesh):
        nodes = mesh.x[:, 0:2].numpy()
        node_to_face_ID = mesh.nodes_faces_ids
        faces = mesh.faces
        distances = np.linalg.norm(nodes - point.numpy(), axis=1)
        closest_node_idx = int(np.argmin(distances))
        possible_faces = node_to_face_ID[closest_node_idx]
        for face_idx in possible_faces:
            face_vertices = faces[:, face_idx]
            triangle_vertices = nodes[face_vertices, 0:2]
            if self._is_point_in_triangle(point, torch.tensor(triangle_vertices, dtype=torch.float)):
                return face_idx
        return None

    def _is_point_in_triangle(self, point, vertices):
        v0 = vertices[2] - vertices[0]
        v1 = vertices[1] - vertices[0]
        v2 = point - vertices[0]
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return False
        inv_denom = 1 / denom
        v = (d11 * d20 - d01 * d21) * inv_denom
        w = (d00 * d21 - d01 * d20) * inv_denom
        u = 1 - v - w
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def _get_best_cross_vector(self, point, previous_direction, mesh, containing_face_idx):
        if containing_face_idx is None:
            return None, mesh
        face_indices = mesh.faces[:, containing_face_idx]
        vertices = mesh.x[face_indices, 0:2]
        ref_vecs = mesh.u[face_indices].to(torch.float)
        bary_coords = self._compute_barycentric_coordinates(point, vertices)
        interpolated_vec = torch.einsum('i,ij->j', bary_coords, ref_vecs)
        interpolated_vec = interpolated_vec / torch.norm(interpolated_vec)
        cross_vectors = []
        base_angle = torch.atan2(interpolated_vec[1], interpolated_vec[0]) / 4
        for i in range(4):
            angle = base_angle + i * (torch.pi / 2)
            v = torch.tensor([torch.cos(angle), torch.sin(angle)], dtype=torch.float)
            cross_vectors.append(v)
        max_dot = -float('inf')
        best_vector = None
        for v in cross_vectors:
            dot = torch.dot(previous_direction, v)
            if dot > max_dot:
                max_dot = dot
                best_vector = v
        return best_vector, mesh

    def _compute_barycentric_coordinates(self, point, vertices):
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = point - vertices[0]
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            raise ValueError("Degenerate triangle.")
        inv_denom = 1 / denom
        v = (d11 * d20 - d01 * d21) * inv_denom
        w = (d00 * d21 - d01 * d20) * inv_denom
        u = 1 - v - w
        return torch.tensor([u, v, w], dtype=torch.float)
