# Graph Report - /home/t1dde/Work/repos/domain_partition  (2026-05-20)

## Corpus Check
- Corpus is ~33,582 words - fits in a single context window. You may not need a graph.

## Summary
- 420 nodes · 637 edges · 37 communities (27 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Mesh Validation|Mesh Validation]]
- [[_COMMUNITY_Main Execution Flow|Main Execution Flow]]
- [[_COMMUNITY_Frame Field Computation|Frame Field Computation]]
- [[_COMMUNITY_Separatrix Generation Backup|Separatrix Generation Backup]]
- [[_COMMUNITY_Frame Field Computation|Frame Field Computation]]
- [[_COMMUNITY_Separatrix Generation|Separatrix Generation]]
- [[_COMMUNITY_Streamline Post-Processing|Streamline Post-Processing]]
- [[_COMMUNITY_Streamline Merging|Streamline Merging]]
- [[_COMMUNITY_Streamline Intersection Splitting|Streamline Intersection Splitting]]
- [[_COMMUNITY_Subdomain Array Extraction|Subdomain Array Extraction]]
- [[_COMMUNITY_Streamline Intersection Splitting|Streamline Intersection Splitting]]
- [[_COMMUNITY_Separatrix Generator Core|Separatrix Generator Core]]
- [[_COMMUNITY_Mesh Data Generation|Mesh Data Generation]]
- [[_COMMUNITY_Separatrix from C0 Nodes|Separatrix from C0 Nodes]]
- [[_COMMUNITY_Streamline Generation|Streamline Generation]]
- [[_COMMUNITY_Streamline Generator v2|Streamline Generator v2]]
- [[_COMMUNITY_NACA Airfoil Geometry|NACA Airfoil Geometry]]
- [[_COMMUNITY_NACA Airfoil Geometry|NACA Airfoil Geometry]]
- [[_COMMUNITY_Transfinite Interpolation|Transfinite Interpolation]]
- [[_COMMUNITY_CSME Evaluation|CSME Evaluation]]
- [[_COMMUNITY_Streamline Initialization|Streamline Initialization]]
- [[_COMMUNITY_Separatrix from Singularity|Separatrix from Singularity]]

## God Nodes (most connected - your core abstractions)
1. `StreamlineSimplificator_v2` - 27 edges
2. `StreamlineSimplificator` - 20 edges
3. `StreamlineSimplificator` - 19 edges
4. `SeparatrixGenerator` - 18 edges
5. `SeparatrixGenerator_v2` - 14 edges
6. `SeparatrixGenerator_v2` - 14 edges
7. `StreamlineGenerator` - 14 edges
8. `StreamlineGenerator_v2` - 13 edges
9. `SeparatrixGenerator` - 13 edges
10. `MeshGenerator` - 12 edges

## Surprising Connections (you probably didn't know these)
- `get_mesh()` --calls--> `StreamlineGenerator_v2`  [INFERRED]
  data_generator.py → tools/streamline_generator_v2.py
- `get_mesh()` --calls--> `StreamlinePostProcessor`  [INFERRED]
  data_generator.py → tools/streamline_post_processor.py
- `main()` --calls--> `StreamlineGenerator`  [INFERRED]
  plot_evaluation.py → tools/streamline_generator.py
- `main()` --calls--> `plot_mesh()`  [INFERRED]
  plot_evaluation.py → tools/plotting_tools.py
- `get_mesh()` --calls--> `MeshCheck`  [INFERRED]
  test.py → tools/check_mesh.py

## Communities (37 total, 10 thin omitted)

### Community 0 - "Mesh Validation"
Cohesion: 0.09
Nodes (12): Check if a quad face is valid (handles airfoil boundaries), Check if a quad is convex, Check if face crosses the airfoil boundary, Find all intersections between streamlines with improved handling, Improved intersection finding with better numerical stability, Check if two line segments may intersect using bounding boxes, Improved mesh generation with better face detection, Create edge index from quad faces (+4 more)

### Community 1 - "Main Execution Flow"
Cohesion: 0.07
Nodes (8): main(), get_new_mesh(), main(), MeshFromFieldgen, plot_faces(), plot_final_mesh(), plot_intersections(), plot_mesh()

### Community 3 - "Separatrix Generation Backup"
Cohesion: 0.14
Nodes (9): Erzeugt Separatrizen von C0-Knoten (Randknoten), die keine regulären Boundary-Kn, Prüft, ob ein Randknoten regulär ist (d.h. die Randkanten mit dem Cross-Vektor ü, Ermittelt den besten Cross-Vektor am gegebenen Punkt basierend auf der vorherige, Berechnet die baryzentrischen Koordinaten eines Punktes in einem Dreieck, Findet das Dreieck, das den gegebenen Punkt enthält, Prüft, ob ein Punkt innerhalb des Meshs liegt, Prüft, ob ein Punkt innerhalb eines Dreiecks liegt, Visualisiert die gefundenen Separatrizen für Debugging (+1 more)

### Community 5 - "Separatrix Generation"
Cohesion: 0.15
Nodes (9): Erzeugt Separatrizen von C0-Knoten (Randknoten), die keine regulären Boundary-Kn, Prüft, ob ein Randknoten regulär ist (d.h. die Randkanten mit dem Cross-Vektor ü, Ermittelt den besten Cross-Vektor am gegebenen Punkt basierend auf der vorherige, Berechnet die baryzentrischen Koordinaten eines Punktes in einem Dreieck, Findet das Dreieck, das den gegebenen Punkt enthält, Prüft, ob ein Punkt innerhalb des Meshs liegt, Prüft, ob ein Punkt innerhalb eines Dreiecks liegt, Visualisiert die gefundenen Separatrizen für Debugging (+1 more)

### Community 6 - "Streamline Post-Processing"
Cohesion: 0.18
Nodes (4): Apply streamline post-processing to handle problematic cases         This modifi, Returns:         Dictionary with:             - 'edges': List of edges where eac, Extracts intersection data into NumPy arrays for subdomain creation.         Imp, StreamlineSimplificator

### Community 7 - "Streamline Merging"
Cohesion: 0.14
Nodes (3): StreamlineMerging, StreamlinePostProcessor, QuadFaceGenerator

### Community 8 - "Streamline Intersection Splitting"
Cohesion: 0.14
Nodes (10): check_bounding_boxes_parallel(), Wrapper for parallel bounding box check., Main method to find intersections and split streamlines., Find all intersection points between streamlines., Find intersection between two splines., Check if splines share endpoints., Optimize to confirm actual intersection., Convert splines back to point arrays. (+2 more)

### Community 9 - "Subdomain Array Extraction"
Cohesion: 0.18
Nodes (3): Returns:         Dictionary with:             - 'edges': List of edges where eac, Extracts intersection data into NumPy arrays for subdomain creation.         Imp, StreamlineSimplificator

### Community 10 - "Streamline Intersection Splitting"
Cohesion: 0.14
Nodes (10): check_bounding_boxes_parallel(), Main method to find intersections and split streamlines., Wrapper for parallel bounding box check., Find all intersection points between streamlines., Find intersection between two splines., Check if splines share endpoints., Optimize to confirm actual intersection., Convert splines back to point arrays. (+2 more)

### Community 12 - "Mesh Data Generation"
Cohesion: 0.19
Nodes (7): extract_mesh_data(), get_mesh(), main(), _mesh_worker(), run_with_timeout(), MeshCheck, QuadMeshGenerator

### Community 14 - "Streamline Generation"
Cohesion: 0.25
Nodes (3): extract_mesh_data(), get_mesh(), StreamlineGenerator

### Community 21 - "Streamline Initialization"
Cohesion: 0.67
Nodes (5): clamp_barycentric_coords(), compute_barycentric_coords(), estimate_singularity_location(), init_streamlines(), interpolate_vector_at_point()

## Knowledge Gaps
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StreamlineSimplificator_v2` connect `Mesh Validation` to `Frame Field Computation`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `MeshGenerator` connect `Frame Field Computation` to `Frame Field Computation`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `StreamlinePostProcessor` connect `Streamline Merging` to `Streamline Intersection Splitting`, `Mesh Data Generation`, `Streamline Post-Processing`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `SeparatrixGenerator` (e.g. with `StreamlineGenerator` and `StreamlineGenerator_v2`) actually correct?**
  _`SeparatrixGenerator` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Erzeugt Separatrizen von C0-Knoten (Randknoten), die keine regulären Boundary-Kn`, `Prüft, ob ein Randknoten regulär ist (d.h. die Randkanten mit dem Cross-Vektor ü`, `Ermittelt den besten Cross-Vektor am gegebenen Punkt basierend auf der vorherige` to the rest of the system?**
  _48 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Mesh Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.08819345661450925 - nodes in this community are weakly interconnected._
- **Should `Main Execution Flow` be split into smaller, more focused modules?**
  _Cohesion score 0.0677361853832442 - nodes in this community are weakly interconnected._