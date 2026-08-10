"""Surface extraction from a Gmsh 2.2 ASCII MSH volume mesh.

Two extraction paths:
- hub/shroud triangle surfaces by geometric region tag (reg_geom: hub=1,
  shroud=2 for the T1 cases), written as binary STL -- the input for the
  unwrap + partition pipeline.
- --include-boundary: the block-structured quad faces of the hexahedral
  core that lie on the hub/shroud cylinders. All exterior hex faces are
  collected (faces referenced by exactly one hex) and split into an inner
  (hub) and outer (shroud) ring by a radius-percentile filter.

TODO: the percentile filter also catches exterior quad faces that are NOT
on the hub/shroud cylinder (e.g. inlet/outlet rim faces); needs a proper
cylinder-distance criterion.
"""

import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HUB_GEOM_TAG = 1
SHROUD_GEOM_TAG = 2

# Gmsh type 5 hexahedron: the six quad faces in local node indices.
_HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def parse_msh(path):
    """Nodes and elements of a Gmsh 2.2 ASCII file.

    Returns (nodes, elements): nodes maps tag -> (x, y, z); elements is a
    list of (etype, phys_tag, geom_tag, node_tags).
    """
    nodes = {}
    elements = []
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("$End"):
                section = None
                continue
            if line.startswith("$"):
                section = line[1:]
                continue
            if section == "Nodes":
                parts = line.split()
                if len(parts) == 4:
                    nodes[int(parts[0])] = (float(parts[1]),
                                            float(parts[2]),
                                            float(parts[3]))
            elif section == "Elements":
                parts = line.split()
                if len(parts) < 5:
                    continue
                etype = int(parts[1])
                num_tags = int(parts[2])
                phys = int(parts[3]) if num_tags >= 1 else 0
                geom = int(parts[4]) if num_tags >= 2 else 0
                node_tags = [int(p) for p in parts[3 + num_tags:]]
                elements.append((etype, phys, geom, node_tags))
    return nodes, elements


def surface_triangles(elements, geom_tag):
    """Triangles of one tagged surface; quads are split into two tris."""
    tris = []
    for etype, _phys, geom, nds in elements:
        if geom != geom_tag:
            continue
        if etype == 2:
            tris.append(nds[:3])
        elif etype == 3:
            tris.append([nds[0], nds[1], nds[2]])
            tris.append([nds[0], nds[2], nds[3]])
    return tris


def write_stl(tris, nodes, path):
    """Binary STL from triangle node-tag lists."""
    def normal(p1, p2, p3):
        u = np.subtract(p2, p1)
        v = np.subtract(p3, p1)
        n = np.cross(u, v)
        length = np.linalg.norm(n)
        return n / length if length > 0 else n

    with open(path, "wb") as f:
        f.write(b"dp3d surface extraction".ljust(80))
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            p = [nodes[t] for t in tri]
            f.write(struct.pack("<fff", *normal(*p)))
            for q in p:
                f.write(struct.pack("<fff", *q))
            f.write(struct.pack("<H", 0))
    print(f"wrote {len(tris)} triangles to {path}")


def extract_hub_shroud(msh_path, out_dir, parts=("hub", "shroud"),
                       parsed=None):
    """Extract the tagged hub/shroud surfaces to <out_dir>/<part>_raw.stl.
    Returns {part: stl_path}."""
    nodes, elements = parsed if parsed else parse_msh(msh_path)
    tags = {"hub": HUB_GEOM_TAG, "shroud": SHROUD_GEOM_TAG}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for part in parts:
        tris = surface_triangles(elements, tags[part])
        if not tris:
            raise ValueError(f"no elements with reg_geom={tags[part]} "
                             f"({part}) in {msh_path}")
        stl = out_dir / f"{part}_raw.stl"
        write_stl(tris, nodes, stl)
        out[part] = stl
    return out


def hex_exterior_quads(elements):
    """Exterior quad faces of the hexahedral core: faces referenced by
    exactly one hex."""
    count = Counter()
    faces = {}
    for etype, _phys, _geom, nds in elements:
        if etype != 5:
            continue
        for loc in _HEX_FACES:
            face = tuple(nds[i] for i in loc)
            key = frozenset(face)
            count[key] += 1
            faces.setdefault(key, face)
    return [faces[k] for k, c in count.items() if c == 1]


def _write_quad_vtk(path, points, quads, node_map):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("dp3d boundary quad extraction\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"CELLS {len(quads)} {5 * len(quads)}\n")
        for q in quads:
            f.write("4 " + " ".join(str(node_map[t]) for t in q) + "\n")
        f.write(f"CELL_TYPES {len(quads)}\n")
        f.write("9\n" * len(quads))  # VTK_QUAD


def extract_boundary_quads(msh_path, out_dir, parts=("hub", "shroud"),
                           inner_pct=25.0, outer_pct=75.0, parsed=None):
    """Block-structured boundary quad meshes of the hex core on the hub
    (inner ring) and shroud (outer ring), written as
    <out_dir>/<part>_boundary_quads.vtk. Returns {part: vtk_path}.

    Selection: radius percentile of the face centroid. Known bug: exterior
    faces off the hub/shroud cylinder can pass this filter (see module
    docstring)."""
    nodes, elements = parsed if parsed else parse_msh(msh_path)
    quads = hex_exterior_quads(elements)
    if not quads:
        raise ValueError(f"no hexahedra in {msh_path} -- "
                         f"--include-boundary needs the hex-core volume mesh")

    centroids = np.array([np.mean([nodes[t] for t in q], axis=0)
                          for q in quads])
    radii = np.hypot(centroids[:, 0], centroids[:, 1])
    thr = {"hub": ("<=", np.percentile(radii, inner_pct)),
           "shroud": (">=", np.percentile(radii, outer_pct))}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for part in parts:
        op, t = thr[part]
        mask = radii <= t if op == "<=" else radii >= t
        sel = [q for q, m in zip(quads, mask) if m]
        used = sorted({t for q in sel for t in q})
        node_map = {t: i for i, t in enumerate(used)}
        points = [nodes[t] for t in used]
        path = out_dir / f"{part}_boundary_quads.vtk"
        _write_quad_vtk(path, points, sel, node_map)
        print(f"wrote {len(sel)} boundary quads ({part}, "
              f"r {op} {t:.4f}) to {path}")
        out[part] = path
    return out
