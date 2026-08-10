
from tools import MeshFromFieldgen,  StreamlineGenerator 
from tools.plotting_tools import *
from tools import StreamlineSimplificator



field_generator  = MeshFromFieldgen("./mesh_out.obj")

mesh            = field_generator.mesh
mesh.u = mesh.frame_field

plot_vector_field(mesh)

streamline      = StreamlineGenerator(mesh)

mesh = streamline.mesh
streamlines_post_processed  = StreamlineSimplificator(mesh)
blocked_mesh                = streamlines_post_processed.quad_mesh

plot_streamlines(streamline.mesh)
plot_intersections(streamlines_post_processed.mesh)
plot_faces(blocked_mesh)



def main():


    fied_generator  = MeshFromFieldgen("./mesh_out.obj")
    mesh            = fied_generator.mesh
    streamline      = StreamlineGenerator(mesh)

    streamlines_post_processed  = StreamlineSimplificator(streamline.mesh)
    blocked_mesh                = streamlines_post_processed.quad_mesh
    
    plot_streamlines(streamline.mesh)
    plot_intersections(streamlines_post_processed.mesh)
    plot_faces(blocked_mesh)

if __name__ == "__main__":
    main()
