import gmsh
import numpy as np
import math
import os


class MshExtractor:
    def __init__(self, input_path, remesh=False, output_path="./temp_remeshed.msh",
                 element_size=4, angle=70, include_boundary=True,
                 force_param=False, curve_angle=180):
        """
        input_path path either to stl or msh file
        performs optional a quasi-structured quad remeshing.
        """

        ext = os.path.splitext(input_path)[1].lower()
        self.remesh = remesh
        self.vertices = None
        self.surfaces = {}  # Dictionary: {surface_tag: faces_array}

        if ext == ".stl":
            self._mesh_stl(input_path, output_path, element_size, angle,
                           include_boundary, force_param, curve_angle)
            load_path = output_path
        elif ext == ".msh":
            if remesh:
                self._remesh_msh(input_path, output_path, element_size,
                                 angle, include_boundary, force_param, curve_angle)
                load_path = output_path
            else:
                load_path = input_path
        else:
            print("File type not supportet, only STL and MSH")
        print(load_path)
        self._load_msh(load_path)

    @staticmethod
    def _mesh_stl(path_to_stl, output_path, element_size, angle,
                  include_boundary, force_param, curve_angle):
        """
        Performs quasi-structured quad remeshing on an STL file.
        """
        gmsh.initialize()
        gmsh.merge(path_to_stl)

        # Classify surfaces to create geometry from STL facets
        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180,
            include_boundary,
            force_param,
            curve_angle * math.pi / 180
        )
        gmsh.model.mesh.createGeometry()

        # Define surface loop and volume for the meshing engine
        surfaces = gmsh.model.getEntities(2)
        tags = [s[1] for s in surfaces]
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(tags)])
        gmsh.model.geo.synchronize()

        # Set element size via a background field
        field_id = gmsh.model.mesh.field.add("MathEval")
        # gmsh.model.mesh.field.setString(
        #     field_id, "F", f"{element_size} + 0.1 * x")
        gmsh.model.mesh.field.setString(field_id, "F", str(element_size))
        gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
        #
        # Set Algorithm 11: Quasi-structured Quad
        gmsh.option.setNumber("Mesh.Algorithm", 11)
        gmsh.model.mesh.generate(2)
        gmsh.write(output_path)
        gmsh.finalize()

    @staticmethod
    def _remesh_msh(path_to_msh, output_path, element_size, angle,
                    include_boundary, force_param, curve_angle):
        """
        Performs quasi-structured quad remeshing on an STL file.
        """
        gmsh.initialize()
        gmsh.open(path_to_msh)

        # Classify surfaces to create geometry from STL facets
        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180,
            include_boundary,
            force_param,
            curve_angle * math.pi / 180
        )
        gmsh.model.mesh.createGeometry()

        # Define surface loop and volume for the meshing engine
        surfaces = gmsh.model.getEntities(2)
        tags = [s[1] for s in surfaces]
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(tags)])
        gmsh.model.geo.synchronize()

        # Set element size via a background field
        field_id = gmsh.model.mesh.field.add("MathEval")
        # gmsh.model.mesh.field.setString(
        #     field_id, "F", f"{element_size} + 0.1 * x")
        gmsh.model.mesh.field.setString(field_id, "F", str(element_size))
        gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
        #
        # Set Algorithm 11: Quasi-structured Quad
        gmsh.option.setNumber("Mesh.Algorithm", 11)
        gmsh.model.mesh.generate(2)
        gmsh.write(output_path)
        gmsh.finalize()

    # @staticmethod
    #
    def _load_msh(self, path):
        gmsh.initialize()
        gmsh.open(path)

        # 1. Nodes (Shared globally)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        self.vertices = coords.reshape(-1, 3)

        tag_to_idx = np.full(int(node_tags.max()) + 1, -1)
        tag_to_idx[node_tags.astype(int)] = np.arange(len(node_tags))

        # 2. Extract faces per Surface
        for _, s_tag in gmsh.model.getEntities(2):
            _, q_tags = gmsh.model.mesh.getElementsByType(3, s_tag)
            if q_tags.size:
                # Store faces for this specific surface tag
                self.surfaces[s_tag] = tag_to_idx[q_tags.astype(
                    int)].reshape(-1, 4)

        gmsh.finalize()
