import gmsh
import numpy as np
from collections import defaultdict
import networkx as nx

OUT_DIR = "/root/repos/block_structured_meshing"

def extract_separatrices(msh_file, name):
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
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 3:
            n_node_per_elem = 4
            for i in range(len(etags)):
                n = enodes[i*n_node_per_elem:(i+1)*n_node_per_elem]
                quads.append([int(x) for x in n])
    
    print(f"[{name}] {len(quads)} quads, {len(node_tags)} nodes")
    
    # build adjacency
    vertex_edges = defaultdict(set)
    edge_quads = defaultdict(list)
    
    for qi, quad in enumerate(quads):
        for i in range(4):
            v1 = quad[i]
            v2 = quad[(i+1)%4]
            edge = tuple(sorted([v1, v2]))
            edge_quads[edge].append(qi)
            vertex_edges[v1].add(v2)
            vertex_edges[v2].add(v1)
    
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
    print(f"[{name}] boundary vertices: {len(boundary_vertices)}")
    
    # compute cyclic order of edges around each vertex
    vertex_order = {}
    for v in vertex_edges:
        neighbors = list(vertex_edges[v])
        if len(neighbors) < 2:
            continue
        
        # get normal vector
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
    
    # build separatrix graph
    G = nx.Graph()
    for path in separatrices:
        for i in range(len(path)-1):
            G.add_edge(path[i], path[i+1])
    
    # add boundary edges
    for v in vertex_edges:
        for nb in vertex_edges[v]:
            edge = tuple(sorted([v, nb]))
            if len(edge_quads[edge]) == 1:
                G.add_edge(v, nb)
    
    # find cycles
    cycles = nx.cycle_basis(G)
    print(f"[{name}] cycles found: {len(cycles)}")
    
    # filter 4-sided cycles
    blocks_4 = []
    for cycle in cycles:
        if len(cycle) == 4:
            blocks_4.append(cycle)
    
    print(f"[{name}] 4-sided blocks: {len(blocks_4)}")
    
    # Write VTK output for visualization
    gmsh.clear()
    
    # Collect all unique vertices
    all_vertices = set()
    for path in separatrices:
        all_vertices.update(path)
    for block in blocks_4:
        all_vertices.update(block)
    
    # Create nodes
    sorted_vertices = sorted(list(all_vertices))
    node_tag_map = {}
    for i, v in enumerate(sorted_vertices):
        tag = i + 1
        node_tag_map[v] = tag
        coord = node_coords[node_map[v]]
        gmsh.model.addDiscreteEntity(0, tag)
        gmsh.model.mesh.addNodes(0, tag, [tag], list(coord))
    
    # Add separatrix lines (element type 1 = 2-node line)
    curve = gmsh.model.addDiscreteEntity(1, 1)
    line_nodes = []
    for path in separatrices:
        for i in range(len(path) - 1):
            line_nodes.extend([node_tag_map[path[i]], node_tag_map[path[i+1]]])
    if line_nodes:
        gmsh.model.mesh.addElementsByType(1, 1, [], line_nodes)
    
    # Add block quads (element type 3 = 4-node quad)
    surf = gmsh.model.addDiscreteEntity(2, 2)
    quad_nodes = []
    for block in blocks_4:
        center = np.zeros(3)
        for v in block:
            center += node_coords[node_map[v]]
        center /= len(block)
        
        corner_angles = []
        for v in block:
            vec = node_coords[node_map[v]] - center
            angle = np.arctan2(vec[1], vec[0])
            if angle < 0:
                angle += 2 * np.pi
            corner_angles.append((angle, v))
        
        corner_angles.sort()
        ordered_corners = [v for _, v in corner_angles]
        quad_nodes.extend([node_tag_map[v] for v in ordered_corners])
    
    if quad_nodes:
        gmsh.model.mesh.addElementsByType(2, 3, [], quad_nodes)
    
    out_vtk = f"{OUT_DIR}/output/T1_9/{name}/{name}_separatrices.vtk"
    gmsh.write(out_vtk)
    print(f"[{name}] saved separatrices to {out_vtk}")
    
    gmsh.finalize()
    
    return separatrices, blocks_4, vertex_order, vertex_edges, edge_quads

if __name__ == "__main__":
    for name in ["hub", "shroud"]:
        msh_file = f"{OUT_DIR}/output/T1_9/{name}/{name}_quad.msh"
        extract_separatrices(msh_file, name)
