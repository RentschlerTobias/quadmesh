import numpy as np


def export_streamlines_to_msh(streamline_dict, output_path="streamlines.msh"):
    """
    Exports a dictionary of streamlines to MSH 2.2 format.
    Input: {surface_tag: [np.array(pts_1), np.array(pts_2), ...]}
    """
    all_points = []
    all_elements = []
    point_offset = 1
    element_id = 1

    # Create unique physical IDs for each surface tag
    surface_tag_map = {tag: i + 1 for i,
                       tag in enumerate(streamline_dict.keys())}

    for surface_tag, lines_list in streamline_dict.items():
        phys_tag = surface_tag_map[surface_tag]

        for points in lines_list:
            # Ensure points is a 2D numpy array (N, 3)
            pts = np.atleast_2d(points)
            if pts.shape[0] < 2:
                continue

            # Handle 2D input (N, 2) -> (N, 3)
            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.zeros((len(pts), 1))])

            start_node_idx = point_offset

            # 1. Add nodes to global list
            for pt in pts:
                all_points.append((point_offset, pt[0], pt[1], pt[2]))
                point_offset += 1

            # 2. Create Line elements connecting the points
            n_pts = len(pts)
            for i in range(n_pts - 1):
                all_elements.append({
                    'id': element_id,
                    'phys': phys_tag,
                    'nodes': [start_node_idx + i, start_node_idx + i + 1]
                })
                element_id += 1

    # Write the MSH 2.2 file
    with open(output_path, 'w') as f:
        # Mesh Format Header
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Physical Names mapping the surface tags
        f.write("$PhysicalNames\n")
        f.write(f"{len(surface_tag_map)}\n")
        for s_tag, p_tag in surface_tag_map.items():
            f.write(f'1 {p_tag} "surface_{s_tag}"\n')
        f.write("$EndPhysicalNames\n")

        # Nodes Section
        f.write("$Nodes\n")
        f.write(f"{len(all_points)}\n")
        for nid, x, y, z in all_points:
            f.write(f"{nid} {x:.10e} {y:.10e} {z:.10e}\n")
        f.write("$EndNodes\n")

        # Elements Section
        f.write("$Elements\n")
        f.write(f"{len(all_elements)}\n")
        for e in all_elements:
            # Format: id type n_tags phys_tag geom_tag node1 node2
            f.write(f"{e['id']} 1 2 {e['phys']} {e['phys']} {
                    e['nodes'][0]} {e['nodes'][1]}\n")
        f.write("$EndElements\n")

    print(f"✓ Exported {len(all_points)} nodes and {
          len(all_elements)} line segments to {output_path}")


def export_streamlines_to_msh_v1(streamlines, output_path="streamlines.msh"):
    """
    Exportiert Stromlinien (Punkte-Arrays) als GMSH .msh Datei (Format 2.2).
    Jede Stromlinie wird als Linienzug (Line-Elemente) gespeichert.
    """

    all_points = []      # Liste aller Punkte (x, y, z)
    # Liste aller Elemente (Linien zwischen aufeinanderfolgenden Punkten)
    all_elements = []
    point_offset = 1     # GMSH ist 1-indiziert
    element_id = 1
    physical_tags = []   # (physical_tag, element_id) Zuordnung

    surface_tag_map = {}  # surface_tag -> physical_tag (int)
    physical_tag_counter = 1

    for surface_tag, data in streamlines.items():
        phys_tag = physical_tag_counter
        surface_tag_map[surface_tag] = phys_tag
        physical_tag_counter += 1

        for points in data['points']:
            points = np.array(points)

            # Sicherstellen dass points shape (N, 3) oder (N, 2) hat
            if points.ndim != 2:
                continue
            if points.shape[1] == 2:
                # 2D -> 3D mit z=0
                points = np.hstack([points, np.zeros((len(points), 1))])

            n_pts = len(points)
            if n_pts < 2:
                continue

            # Punkte hinzufügen
            start_idx = point_offset
            for pt in points:
                all_points.append((point_offset, pt[0], pt[1], pt[2]))
                point_offset += 1

            # Linien-Elemente zwischen aufeinanderfolgenden Punkten
            for i in range(n_pts - 1):
                all_elements.append({
                    'id': element_id,
                    'type': 1,          # Typ 1 = 2-Knoten Linie
                    'phys_tag': phys_tag,
                    'geom_tag': phys_tag,
                    'nodes': [start_idx + i, start_idx + i + 1]
                })
                physical_tags.append((phys_tag, element_id))
                element_id += 1

    # .msh Datei schreiben
    with open(output_path, 'w') as f:

        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        # Physical Names (optional, aber hilfreich)
        f.write("$PhysicalNames\n")
        f.write(f"{len(surface_tag_map)}\n")
        for surface_tag, phys_tag in surface_tag_map.items():
            f.write(f'1 {phys_tag} "surface_{surface_tag}"\n')
        f.write("$EndPhysicalNames\n")

        # Knoten
        f.write("$Nodes\n")
        f.write(f"{len(all_points)}\n")
        for node_id, x, y, z in all_points:
            f.write(f"{node_id} {x:.10e} {y:.10e} {z:.10e}\n")
        f.write("$EndNodes\n")

        # Elemente
        f.write("$Elements\n")
        f.write(f"{len(all_elements)}\n")
        for elem in all_elements:
            nodes_str = " ".join(str(n) for n in elem['nodes'])
            # Format: elem_id  elem_type  n_tags  phys_tag  geom_tag  node1  node2
            f.write(f"{elem['id']} {elem['type']} 2 {
                    elem['phys_tag']} {elem['geom_tag']} {nodes_str}\n")
        f.write("$EndElements\n")

    print(f"✓ Exportiert: {len(all_points)} Punkte, {
          len(all_elements)} Linien-Elemente → {output_path}")
    return output_path
