import gmsh
import numpy as np
from collections import defaultdict
import networkx as nx

OUT_DIR = "/root/repos/block_structured_meshing"

def extract_and_coarsen_blocks(msh_file, name, out_file, min_block_size=0.5):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    gmsh.open(msh_file)
    
    # get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}
    
    # get quads
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    quads = []
    quad_nodes_list = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 3:
            n_node_per_elem = 4
            for i in range(len(etags)):
                n = enodes[i*n_node_per_elem:(i+1)*n_node_per_elem]
                quads.append([int(x) for x in n])
                quad_nodes_list.append(list(n))
    
    print(f"[{name}] {len(quads)} quads, {len(node_tags)} nodes")
    
    # build adjacency
    vertex_edges = defaultdict(set)
    edge_quads = defaultdict(list)
    quad_edges = []
    
    for qi, quad in enumerate(quads):
        q_edges = []
        for i in range(4):
            v1 = quad[i]
            v2 = quad[(i+1)%4]
            edge = tuple(sorted([v1, v2]))
            edge_quads[edge].append(qi)
            vertex_edges[v1].add(v2)
            vertex_edges[v2].add(v1)
            q_edges.append(edge)
        quad_edges.append(q_edges)
    
    # compute valence
    valence = {}
    for v in vertex_edges:
        valence[v] = len(vertex_edges[v])
    
    # identify boundary vertices
    boundary_vertices = set()
    for v in vertex_edges:
        for nb in vertex_edges[v]:
            edge = tuple(sorted([v, nb]))
            if len(edge_quads[edge]) == 1:
                boundary_vertices.add(v)
                break
    
    # identify singular vertices
    singular = set()
    for v in vertex_edges:
        if v not in boundary_vertices:
            if valence[v] != 4:
                singular.add(v)
    
    print(f"[{name}] singular vertices: {len(singular)}")
    
    # compute cyclic order
    vertex_order = {}
    for v in vertex_edges:
        neighbors = list(vertex_edges[v])
        if len(neighbors) < 2:
            continue
        
        normals = []
        for nb in neighbors:
            edge = tuple(sorted([v, nb]))
            for qi in edge_quads[edge]:
                quad = quads[qi]
                idx = [node_map[x] for x in quad]
                pts = node_coords[idx]
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[1]
                n = np.cross(v1, v2)
                if np.linalg.norm(n) > 1e-10:
                    n = n / np.linalg.norm(n)
                    normals.append(n)
        
        if len(normals) == 0:
            continue
        
        normal = np.mean(normals, axis=0)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        v_pos = node_coords[node_map[v]]
        projected = []
        for nb in neighbors:
            nb_pos = node_coords[node_map[nb]]
            vec = nb_pos - v_pos
            vec = vec - np.dot(vec, normal) * normal
            if np.linalg.norm(vec) > 1e-10:
                projected.append((nb, vec))
        
        if len(projected) < 2:
            continue
        
        ref = projected[0][1]
        ref = ref / np.linalg.norm(ref)
        
        angles = []
        for nb, vec in projected:
            vec = vec / np.linalg.norm(vec)
            angle = np.arctan2(np.dot(np.cross(ref, vec), normal), np.dot(ref, vec))
            if angle < 0:
                angle += 2 * np.pi
            angles.append((angle, nb))
        
        angles.sort()
        vertex_order[v] = [nb for _, nb in angles]
    
    # trace separatrices
    separatrices = []
    visited_paths = set()
    
    for v0 in singular:
        if v0 not in vertex_order:
            continue
        
        order = vertex_order[v0]
        n_edges = len(order)
        
        for i in range(n_edges):
            v1 = order[i]
            
            path = [v0, v1]
            current = v1
            prev = v0
            
            max_iter = 1000
            iter_count = 0
            while iter_count < max_iter:
                iter_count += 1
                
                if current in singular:
                    break
                
                if current not in vertex_order:
                    break
                
                order_curr = vertex_order[current]
                n_curr = len(order_curr)
                
                if n_curr < 2:
                    break
                
                if prev not in order_curr:
                    break
                
                idx = order_curr.index(prev)
                opposite_idx = (idx + n_curr // 2) % n_curr
                next_v = order_curr[opposite_idx]
                
                if next_v == path[-1]:
                    break
                
                path.append(next_v)
                prev = current
                current = next_v
                
                if current in path[:-1]:
                    break
            
            path_tuple = tuple(sorted([tuple(path), tuple(reversed(path))])[0])
            if path_tuple not in visited_paths:
                visited_paths.add(path_tuple)
                separatrices.append(path)
    
    print(f"[{name}] separatrices: {len(separatrices)}")
    
    # Filter separatrices by length
    # Compute length of each separatrix
    separatrix_lengths = []
    for path in separatrices:
        length = 0
        for i in range(len(path)-1):
            v1 = path[i]
            v2 = path[i+1]
            idx1 = node_map[v1]
            idx2 = node_map[v2]
            p1 = node_coords[idx1]
            p2 = node_coords[idx2]
            length += np.linalg.norm(p2 - p1)
        separatrix_lengths.append((length, path))
    
    # Sort by length
    separatrix_lengths.sort(reverse=True)
    
    # Keep only longest separatrices (top N)
    # For ~20 blocks, we need ~10-16 separatrices depending on surface size
    if name == "hub":
        n_keep = min(len(separatrix_lengths), 16)
    elif name == "shroud":
        n_keep = min(len(separatrix_lengths), 14)
    else:
        n_keep = min(len(separatrix_lengths), 16)
    filtered_separatrices = [path for _, path in separatrix_lengths[:n_keep]]
    
    print(f"[{name}] filtered separatrices: {len(filtered_separatrices)}")
    
    # Build separatrix graph
    G = nx.Graph()
    for path in filtered_separatrices:
        for i in range(len(path)-1):
            G.add_edge(path[i], path[i+1])
    
    # Add boundary edges
    for v in vertex_edges:
        for nb in vertex_edges[v]:
            edge = tuple(sorted([v, nb]))
            if len(edge_quads[edge]) == 1:
                G.add_edge(v, nb)
    
    # Find cycles
    cycles = nx.cycle_basis(G)
    print(f"[{name}] cycles found: {len(cycles)}")
    
    # Filter 4-sided cycles
    blocks_4 = []
    for cycle in cycles:
        if len(cycle) == 4:
            blocks_4.append(cycle)
    
    print(f"[{name}] 4-sided blocks: {len(blocks_4)}")
    
    # Create coarse block mesh
    gmsh.clear()
    
    # Collect all vertices on block boundaries
    all_boundary_vertices = set()
    for block in blocks_4:
        for v in block:
            all_boundary_vertices.add(v)
    
    all_boundary_vertices = sorted(list(all_boundary_vertices))
    node_tag_map = {}
    for i, v in enumerate(all_boundary_vertices):
        tag = i + 1
        node_tag_map[v] = tag
        idx = node_map[v]
        coord = node_coords[idx]
        gmsh.model.addDiscreteEntity(0, tag)
        gmsh.model.mesh.addNodes(0, tag, [tag], list(coord))
    
    # Add quads for blocks
    surf = gmsh.model.addDiscreteEntity(2, 2)
    quad_nodes = []
    for block in blocks_4:
        # Compute block center
        center = np.zeros(3)
        for v in block:
            idx = node_map[v]
            center += node_coords[idx]
        center /= len(block)
        
        # Find 4 corner vertices
        # corners are boundary vertices with 2 boundary edges
        block_boundary_edges = set()
        for v in block:
            for nb in vertex_edges[v]:
                edge = tuple(sorted([v, nb]))
                if edge in [(tuple(sorted([block[i], block[(i+1)%4]]))) for i in range(4)]:
                    block_boundary_edges.add(edge)
        
        # Order corners by angle from center
        corner_angles = []
        for v in block:
            idx = node_map[v]
            vec = node_coords[idx] - center
            angle = np.arctan2(vec[1], vec[0])
            if angle < 0:
                angle += 2 * np.pi
            corner_angles.append((angle, v))
        
        corner_angles.sort()
        ordered_corners = [v for _, v in corner_angles]
        
        quad_nodes.extend([node_tag_map[v] for v in ordered_corners])
    
    if quad_nodes:
        gmsh.model.mesh.addElementsByType(2, 3, [], quad_nodes)
    
    # Write both formats
    out_msh = out_file + '.msh'
    gmsh.write(out_msh)
    print(f"[{name}] saved coarse blocks to {out_msh}")
    
    out_vtk = out_file + '.vtk'
    gmsh.write(out_vtk)
    print(f"[{name}] saved coarse blocks to {out_vtk}")
    
    gmsh.finalize()

if __name__ == "__main__":
    for name in ["hub", "shroud"]:
        msh_file = f"{OUT_DIR}/output/T1_9/{name}/{name}_quad.msh"
        out_file = f"{OUT_DIR}/output/T1_9/{name}/{name}_blocks"
        extract_and_coarsen_blocks(msh_file, name, out_file, min_block_size=0.5)
