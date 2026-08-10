import meshio
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np


def plot_mesh(nodes, faces, output_file="./figures/quad_mesh.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    figsize = (5, 5)
    plt.figure(figsize=figsize)

    for face in faces.T:
        coords = nodes[face]  # shape (4, 2)
        color = np.random.rand(3,)  # Random RGB color for each face
        plt.fill(coords[:, 0], coords[:, 1], color=color,
                 edgecolor='gray', linewidth=0.5)

    plt.axis('off')
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)


def streamlines_to_html(streamlines, output_path="streamlines.html"):
    fig = go.Figure()

    for streamline in streamlines:
        x, y, z = streamline[:, 0], streamline[:, 1], streamline[:, 2]

        # zufällige Farbe für jede Stromlinie
        color = np.random.randint(0, 255, size=3)
        color = f'rgb({color[0]},{color[1]},{color[2]})'

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=4)
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        title="3D Stromlinien"
    )

    # Speichern als HTML
    fig.write_html(output_path)
    print(f"Plot gespeichert als: {output_path}")


def block_structure_to_html(graph_data, output_path="blocking.html"):
    """
    Input: dict -> keys are surface tags, values contain 'vertices', 'edges', 'edge_to_streamline', 'streamlines'

    Output: None (saves HTML file)

    Plots 3D visualization showing:
        - Streamlines as colored curves
        - Vertices as points at start/end locations
        - Edges as lines connecting vertices with matching streamline colors
    """

    fig = go.Figure()

    # Assign unique color to each streamline for consistency between streamline and edge
    streamline_colors = {}  # (surface_tag, streamline_idx) -> color string

    for surface_tag, data in graph_data.items():
        vertices = data['vertices']
        edges = data['edges']
        streamlines = data['streamlines']

        # Generate unique colors for each streamline in this surface
        for sl_idx in range(len(streamlines)):
            color = np.random.randint(0, 255, size=3)
            color_str = f'rgb({color[0]},{color[1]},{color[2]})'
            streamline_colors[(surface_tag, sl_idx)] = color_str

        # Plot streamlines with assigned colors
        for sl_idx, streamline in enumerate(streamlines):
            x, y, z = streamline[:, 0], streamline[:, 1], streamline[:, 2]
            color_str = streamline_colors[(surface_tag, sl_idx)]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color_str, width=4),
                name=f'{surface_tag} streamline {
                    sl_idx}' if sl_idx == 0 else None
            ))

        # Plot edges with matching colors (use color of first associated streamline)
        for edge_idx, edge in enumerate(edges):
            start_vertex = vertices[edge[0]]
            end_vertex = vertices[edge[1]]

            # Get the first streamline index associated with this edge to determine color
            sl_indices = data['edge_to_streamline'].get(edge_idx, [])
            if sl_indices:
                first_sl_idx = sl_indices[0]
                edge_color = streamline_colors.get(
                    (surface_tag, first_sl_idx), 'gray')
            else:
                edge_color = 'gray'

            fig.add_trace(go.Scatter3d(
                x=[start_vertex[0], end_vertex[0]],
                y=[start_vertex[1], end_vertex[1]],
                z=[start_vertex[2], end_vertex[2]],
                mode='lines',
                line=dict(color=edge_color, width=4),
                name=f'{surface_tag} edge' if edge_idx == 0 else None
            ))

        # Plot vertices as points
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            mode='markers',
            marker=dict(size=8, color='black'),
            name=f'{surface_tag} vertices' if surface_tag == list(graph_data.keys())[
                0] else None
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        title="3D Stromlinien"
    )

    # Speichern als HTML
    fig.write_html(output_path)
    print(f"Plot gespeichert als: {output_path}")


def interpolate_quad_surface(sl_01, sl_12, sl_23, sl_30, n_u=20, n_v=20):
    """
    Bilineare Surface Interpolation aus 4 Randkurven.

    sl_01: Stromlinie von v0 -> v1 (shape: n_points, 3)
    sl_12: Stromlinie von v1 -> v2
    sl_23: Stromlinie von v2 -> v3  (wird umgekehrt: v3->v2 = gegenüber von sl_01)
    sl_30: Stromlinie von v3 -> v0  (wird umgekehrt: v0->v3 = gegenüber von sl_12)

    Gibt X, Y, Z Gitter zurück (je n_u x n_v) für go.Surface
    """
    from scipy.interpolate import interp1d

    def resample(sl, n):
        """Stromlinie auf n gleichmäßige Punkte resamplen"""
        sl = np.asarray(sl)
        n_points = len(sl)
        t_old = np.linspace(0, 1, n_points)
        t_new = np.linspace(0, 1, n)
        if n_points >= 4:
            interp_kind = 'cubic'
        elif n_points > 1:
            interp_kind = 'linear'
        else:
            # Falls nur ein Punkt existiert (sollte nicht passieren), diesen einfach kacheln
            return np.tile(sl[0], (n, 1))

        f = interp1d(t_old, sl, axis=0, kind=interp_kind)
        return f(t_new)    # Alle 4 Kanten auf gleiche Auflösung bringen

    # Konvention: u läuft entlang sl_01/sl_23, v läuft entlang sl_12/sl_30
    C0 = resample(sl_01, n_u)   # v=0 Kante: v0 -> v1
    # v=1 Kante: v3 -> v2 (Gegenüber, gleiche Richtung wie C0)
    C1 = resample(sl_23, n_u)
    D0 = resample(sl_30, n_v)   # u=0 Kante: v0 -> v3
    D1 = resample(sl_12, n_v)   # u=1 Kante: v1 -> v2

    # surface patch: P(u,v) = (1-v)*C0(u) + v*C1(u)
    #                      + (1-u)*D0(v) + u*D1(v)
    #                      - bilineare Korrektur der 4 Ecken
    u = np.linspace(0, 1, n_u)
    v = np.linspace(0, 1, n_v)
    uu, vv = np.meshgrid(u, v)  # beide (n_v, n_u)

    # Ecken
    P00 = C0[0]   # v0
    P10 = C0[-1]  # v1
    P01 = C1[0]   # v3
    P11 = C1[-1]  # v2

    # Interpolation: Broadcasting über Grid
    # C0(u): shape (n_u, 3) -> broadcast zu (n_v, n_u, 3)
    # vv:    shape (n_v, n_u) -> (n_v, n_u, 1)
    surf = (
        (1 - vv[:, :, None]) * C0[None, :, :]       # (1-v) * C0(u)
        + vv[:, :, None] * C1[None, :, :]  # v  * C1(u)
        + (1 - uu[:, :, None]) * D0[:, None, :]        # (1-u) * D0(v)
        + uu[:, :, None] * D1[:, None, :]  # u  * D1(v)
        - (1 - uu[:, :, None]) * (1 - vv[:, :, None]) *
        P00  # bilineare Eckenkorrektur
        - uu[:, :, None] * (1 - vv[:, :, None]) * P10
        - (1 - uu[:, :, None]) * vv[:, :, None] * P01
        - uu[:, :, None] * vv[:, :, None] * P11
    )

    return surf[:, :, 0], surf[:, :, 1], surf[:, :, 2]  # X, Y, Z je (n_v, n_u)


def faces_to_html(graph_data, output_path="blocking_faces.html", n_u=20, n_v=20):
    """
    Generates interactive html plot of the quadrilateral Block-Structure
    The faces can be interpolated for better visualization using equidistant bilineare interpolation
    specified by n_u, n_v 
    no interpolation if n_u= None, n_v= None, just linear connection of the vertices
    """
    fig = go.Figure()

    for surface_tag, data in graph_data.items():
        vertices = data['vertices']
        faces = data['faces']               # (4, n_faces)
        vertex_pair_to_sl = data['edge_to_streamline']  # (va,vb) -> streamline

        n_faces = faces.shape[1]

        for face_idx in range(n_faces):
            v_ids = faces[:, face_idx]  # [v0, v1, v2, v3]
            v0, v1, v2, v3 = int(v_ids[0]), int(
                v_ids[1]), int(v_ids[2]), int(v_ids[3])

            r, g, b = np.random.randint(50, 220, size=3)
            face_color = f'rgb({r},{g},{b})'

            # Stromlinien der 4 Kanten holen (Richtung beachten!)
            sl_01 = vertex_pair_to_sl.get((v0, v1))  # v0 -> v1
            sl_12 = vertex_pair_to_sl.get((v1, v2))  # v1 -> v2
            # v3 -> v2 (Gegenüber von sl_01, gleiche u-Richtung)
            sl_23 = vertex_pair_to_sl.get((v3, v2))
            sl_30 = vertex_pair_to_sl.get((v0, v3))  # v0 -> v3

            if any(sl is None for sl in [sl_01, sl_12, sl_23, sl_30]):
                missing = [(va, vb) for (va, vb), sl in [((v0, v1), sl_01), ((
                    v1, v2), sl_12), ((v3, v2), sl_23), ((v0, v3), sl_30)] if sl is None]
                print(f"WARNUNG: Fehlende Stromlinien in Face {
                      face_idx}: {missing}")
                continue

            # surface interpolieren
            try:
                X, Y, Z = interpolate_quad_surface(
                    sl_01, sl_12, sl_23, sl_30, n_u=n_u, n_v=n_v)

                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale=[[0, face_color], [1, face_color]],
                    showscale=False,
                    opacity=0.6,
                    name=f'{surface_tag} face {face_idx}',
                    showlegend=(face_idx == 0),
                    contours=dict(
                        x=dict(show=False),
                        y=dict(show=False),
                        z=dict(show=False),
                    )
                ))
            except Exception as e:
                print(e)
                print('surface interpoltion failed')
                # Kanten als Stromlinien drüber zeichnen
            for (va, vb) in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]:
                sl = vertex_pair_to_sl.get((va, vb))
                if sl is not None:
                    fig.add_trace(go.Scatter3d(
                        x=sl[:, 0], y=sl[:, 1], z=sl[:, 2],
                        mode='lines',
                        line=dict(color=face_color, width=3),
                        showlegend=False
                    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900, height=900,
        title="3D Block-Structure - Interpolated Faces: "
    )

    fig.write_html(output_path)
    print(f"Plot gespeichert als: {output_path}")


def surfaces_to_html(graph_data, output_name="blocking_faces_independent_surfaces", n_u=20, n_v=20):

    for surface_tag, data in graph_data.items():

        fig = go.Figure()
        vertices = data['vertices']
        faces = data['faces']               # (4, n_faces)
        vertex_pair_to_sl = data['edge_to_streamline']  # (va,vb) -> streamline

        n_faces = faces.shape[1]

        for face_idx in range(n_faces):
            v_ids = faces[:, face_idx]  # [v0, v1, v2, v3]
            v0, v1, v2, v3 = int(v_ids[0]), int(
                v_ids[1]), int(v_ids[2]), int(v_ids[3])

            r, g, b = np.random.randint(50, 220, size=3)
            face_color = f'rgb({r},{g},{b})'

            # Stromlinien der 4 Kanten holen (Richtung beachten!)
            sl_01 = vertex_pair_to_sl.get((v0, v1))  # v0 -> v1
            sl_12 = vertex_pair_to_sl.get((v1, v2))  # v1 -> v2
            # v3 -> v2 (Gegenüber von sl_01, gleiche u-Richtung)
            sl_23 = vertex_pair_to_sl.get((v3, v2))
            sl_30 = vertex_pair_to_sl.get((v0, v3))  # v0 -> v3

            if any(sl is None for sl in [sl_01, sl_12, sl_23, sl_30]):
                missing = [(va, vb) for (va, vb), sl in [((v0, v1), sl_01), ((
                    v1, v2), sl_12), ((v3, v2), sl_23), ((v0, v3), sl_30)] if sl is None]
                print(f"WARNUNG: Fehlende Stromlinien in Face {
                      face_idx}: {missing}")
                continue

            # surface interpolieren
            try:
                X, Y, Z = interpolate_quad_surface(
                    sl_01, sl_12, sl_23, sl_30, n_u=n_u, n_v=n_v)

                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale=[[0, face_color], [1, face_color]],
                    showscale=False,
                    opacity=0.6,
                    name=f'{surface_tag} face {face_idx}',
                    showlegend=(face_idx == 0),
                    contours=dict(
                        x=dict(show=False),
                        y=dict(show=False),
                        z=dict(show=False),
                    )
                ))
            except Exception as e:
                print(e)
                print('surface interpoltion failed')
                # Kanten als Stromlinien drüber zeichnen
            for (va, vb) in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]:
                sl = vertex_pair_to_sl.get((va, vb))
                if sl is not None:
                    fig.add_trace(go.Scatter3d(
                        x=sl[:, 0], y=sl[:, 1], z=sl[:, 2],
                        mode='lines',
                        line=dict(color=face_color, width=3),
                        showlegend=False
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900, height=900,
            title="3D Block-Struktur: Interpolierte Faces"
        )
        output_path = f'./figures/{output_name}_s{surface_tag}.html'
        fig.write_html(output_path)
        print(f"Plot gespeichert als: {output_path}")


def surfaces_to_msh(graph_data, output_name="blocking_geometry", n_u=20, n_v=20):
    """
    Konvertiert die einzelne Flaechen  in eine .msh Datei.
    Jedes Face (Quad_Block) wird als diskretes Quad-Gitter (äquidistante Transfinite Interpolation mit n_u x n_v Elementen) exportiert.
    """

    all_points = []
    all_cells = []
    point_offset = 0

    for surface_tag, data in graph_data.items():
        faces = data['faces']
        vertex_pair_to_sl = data['edge_to_streamline']
        n_faces = faces.shape[1]

        for face_idx in range(n_faces):
            v_ids = faces[:, face_idx]
            v0, v1, v2, v3 = int(v_ids[0]), int(
                v_ids[1]), int(v_ids[2]), int(v_ids[3])

            # Stromlinien holen
            sl_01 = vertex_pair_to_sl.get((v0, v1))
            sl_12 = vertex_pair_to_sl.get((v1, v2))
            sl_23 = vertex_pair_to_sl.get((v3, v2))
            sl_30 = vertex_pair_to_sl.get((v0, v3))

            if any(sl is None for sl in [sl_01, sl_12, sl_23, sl_30]):
                continue

            try:
                if n_u is None and n_v is None:
                    # Extract endpoints from streamlines as vertex coordinates
                    p0 = np.array([sl_01[0, 0], sl_01[0, 1], sl_01[0, 2]])
                    p1 = np.array([sl_01[-1, 0], sl_01[-1, 1], sl_01[-1, 2]])
                    p2 = np.array([sl_12[-1, 0], sl_12[-1, 1], sl_12[-1, 2]])
                    p3 = np.array([sl_30[-1, 0], sl_30[-1, 1], sl_30[-1, 2]])

                    face_points = np.stack([p0, p1, p2, p3])  # Shape: (4, 3)
                    all_points.append(face_points)

                    # Single quad cell referencing the four local points
                    single_quad = [
                        [point_offset + 0, point_offset + 1, point_offset + 2, point_offset + 3]]
                    all_cells.append(("quad", np.array(single_quad)))
                    point_offset += len(face_points)

                else:
                    # surface interpolation
                    X, Y, Z = interpolate_quad_surface(
                        sl_01, sl_12, sl_23, sl_30, n_u=n_u, n_v=n_v)

                    # Form: (n_u * n_v, 3)
                    patch_points = np.stack(
                        [X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                    all_points.append(patch_points)

                    # Indizes für die Quad-Elemente innerhalb des Patches berechnen
                    quads = []
                    for j in range(n_v - 1):
                        for i in range(n_u - 1):
                            # Lokale Indizes im Patch-Gitter
                            p0 = point_offset + (j * n_u + i)
                            p1 = point_offset + (j * n_u + i + 1)
                            p2 = point_offset + ((j + 1) * n_u + i + 1)
                            p3 = point_offset + ((j + 1) * n_u + i)
                            quads.append([p0, p1, p2, p3])

                    all_cells.append(("quad", np.array(quads)))
                    point_offset += len(patch_points)

            except Exception as e:
                print(f"Fehler bei Face {face_idx}: {e}")

    # Alle Daten zusammenführen
    if not all_points:
        print("Keine Geometriedaten zum Exportieren vorhanden.")
        return

    final_points = np.concatenate(all_points, axis=0)

    # Mesh-Objekt erstellen
    mesh = meshio.Mesh(
        points=final_points,
        cells=all_cells
    )

    full_output_path = f"./{output_name}.msh"
    # Binärformat 2.2 ist hochkompatibel
    mesh.write(full_output_path, file_format="gmsh22")
    print(f"MSH-Datei erfolgreich gespeichert: {full_output_path}")
