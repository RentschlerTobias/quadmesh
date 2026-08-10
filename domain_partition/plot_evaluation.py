
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator, Transfinite_Interpolation
from tools.plotting_tools import *
import torch


mesh = torch.load("./saved_meshes/good_mesh.pt", weights_only=False)

def main():

    file = 'test'
    
    airfoil                 = NACA_airfoil()

    mesh_init_triangulation = MeshGenerator(airfoil, quadMesh=False, lc=0.075)
    frameField              = FrameField(mesh_init_triangulation.mesh)

    streamline              = StreamlineGenerator(frameField.mesh)

    post_processing         = StreamlineSimplificator(streamline.mesh)
    blocked_mesh            = post_processing.quad_mesh

    transfinite_interpolation = Transfinite_Interpolation(blocked_mesh)
    quad_mesh = transfinite_interpolation.quad_mesh

    plot_mesh(quad_mesh, output_file=f"./figures/quad_mesh_transfinite_{file}.png")

    mesh = post_processing.mesh
    plot_vector_field(mesh, init=True, output_file=f"./figures/vector_field_init_{file}.png")
    plot_vector_field(mesh, init=False, output_file=f"./figures/vector_field_propagated_{file}.png")
    plot_cross_field(mesh, init=True, output_file=f"./figures/cross_field_init_{file}.png")
    plot_cross_field(mesh, init=False, output_file=f"./figures/cross_field_propagated_{file}.png")
    plot_singularities(mesh, output_file=f"./figures/mesh_singularities_{file}.png")
    plot_streamlines(mesh, output_file=f"./figures/streamlines_colored_{file}.png", colored=True)
    plot_intersections(mesh, output_file=f"./figures/intersections_{file}.png")
    plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")


if __name__ == "__main__":
    main()
