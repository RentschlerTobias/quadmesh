from torch_geometric import edge_index
from torch_geometric.data import Data
from .streamline_merging import StreamlineMerging
from .streamline_intersection_splitter import StreamlineIntersectionSplitter
from .streamlines_to_quad_faces import QuadFaceGenerator


class StreamlinePostProcessor:

    def __init__(self, mesh: Data, verbose: bool = True):

        self.mesh                       = mesh
        streamlineMerging               = StreamlineMerging(self.mesh, verbose=verbose)
        new_streamlines                 = streamlineMerging.new_streamlines

        splitter                        = StreamlineIntersectionSplitter(offset_boundingBox=0.05, num_samples=5)
        updated_streamlines             = splitter.process_streamlines(new_streamlines)
        faceGenerator                   = QuadFaceGenerator(updated_streamlines)

        faces, edge_to_streamline, edge_index, nodes = faceGenerator.get_data()
        self.block_mesh = Data(x=nodes, edge_index=edge_index, faces=faces, edge_to_streamline=edge_to_streamline)
