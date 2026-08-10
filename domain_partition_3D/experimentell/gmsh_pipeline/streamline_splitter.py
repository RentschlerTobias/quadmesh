import numpy as np
from collections import Counter


def get_global_counts(surface_dict):
    # Count occurrences globally, but only once per loop-ring
    counts = Counter()
    for streamlines in surface_dict.values():
        for sl in streamlines:
            if len(sl) < 2:
                continue
            is_loop = np.array_equal(sl[0], sl[-1])
            # For loops, exclude the duplicate end point from the global count
            points_to_count = sl[:-1] if is_loop else sl
            counts.update(map(tuple, points_to_count))
    return counts


def normalize_loop(sl, global_counts):
    is_loop = np.array_equal(sl[0], sl[-1])
    if not is_loop:
        return sl

    # Local count: how often is the point part of THIS specific loop?
    local_points = [tuple(p) for p in sl[:-1]]
    local_counts = Counter(local_points)

    for i in range(len(sl) - 1):
        point_tpl = local_points[i]
        # Real intersection: someone else uses this point too
        if global_counts[point_tpl] > local_counts[point_tpl]:
            # Rotate loop so it starts/ends at this real intersection
            ring = sl[:-1]
            temp = np.vstack([ring[i:], ring[:i]])
            return np.vstack([temp, temp[0]])

    return sl


def split_streamlines(surface_dict):
    global_counts = get_global_counts(surface_dict)
    new_surface_dict = {}

    for surf_name, streamlines in surface_dict.items():
        processed = []
        for sl in streamlines:
            if len(sl) < 2:
                continue

            sl = normalize_loop(sl, global_counts)

            # Determine if current sl is a loop to handle local counting
            is_loop = np.array_equal(sl[0], sl[-1])
            relevant_points = sl[:-1] if is_loop else sl
            local_counts = Counter(map(tuple, relevant_points))

            # Identify split indices
            split_indices = [0]
            for i in range(1, len(sl) - 1):
                point_tpl = tuple(sl[i])
                # CRITICAL: Only split if global count exceeds local count
                # This prevents splitting at a loop's arbitrary start/end point
                if global_counts[point_tpl] > local_counts.get(point_tpl, 0):
                    split_indices.append(i)
            split_indices.append(len(sl) - 1)

            # Unique sorted indices to prevent empty segments
            split_indices = sorted(list(set(split_indices)))

            for start, end in zip(split_indices[:-1], split_indices[1:]):
                if start < end:
                    processed.append(np.array(sl[start: end + 1]))

        new_surface_dict[surf_name] = processed

    return new_surface_dict
