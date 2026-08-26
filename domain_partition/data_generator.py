
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator_v2, StreamlinePostProcessor
from tools import QuadMeshGenerator
from tools import MeshCheck, QuadPartitionValidator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import sys
import multiprocessing as mp
import time
import tempfile
import argparse
from tqdm import tqdm

def get_mesh():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

    try:
        airfoil                     = NACA_airfoil()
        random_lc                   = 0.04 + 0.02 * np.random.rand()
        mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
        frameField                  = FrameField(mesh_gen.mesh)
        streamline                  = StreamlineGenerator_v2(frameField.mesh)
        streamlines_post_processed  = StreamlinePostProcessor(streamline.mesh)
        blocked_mesh                = streamlines_post_processed.block_mesh
        tri_mesh                    = streamlines_post_processed.mesh

        # --- Pre-Filter: Quad Partition Validator (Phase 2: strict=True) ---
        validator                   = QuadPartitionValidator(blocked_mesh, tri_mesh, strict=True)
        if not validator.is_valid():
            print('failed: blocked mesh invalid (pre-filter)')
            print('\n'.join(validator.diagnostics()))
            return None
        qs = validator.quality_score()
        print(f"Pre-filter quality: SJ_min={qs.get('scaled_jacobian_min', -1):.3f}, "
              f"angle=[{qs.get('min_interior_angle', -1):.1f}, {qs.get('max_interior_angle', -1):.1f}], "
              f"aspect={qs.get('edge_length_ratio_max', -1):.2f}")
        # --------------------------------------------------------------------

        transfiniteInterpolator     = QuadMeshGenerator(blocked_mesh)
        quad_mesh                   = transfiniteInterpolator.transfinite_mesh
        mesh_check                  = MeshCheck(tri_mesh, quad_mesh, tol=1e-3)
        success                     = mesh_check.is_valid
        print(f'!!! \n area difference: \n {mesh_check.quad_area - mesh_check.tri_area}\n !!!')
        if success == True:
            mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)
            print('succssess')
            return mesh
        else:
            print('failed')
            return None
    except Exception as e:
        import traceback
        print(f'\n domain partition failed: {type(e).__name__}: {e} \n')
        traceback.print_exc()

def _mesh_worker(queue, idx, quiet=False):
    # Silence ALL child output (Python prints + gmsh/numba C-level stdout/stderr)
    # at the file-descriptor level so the tqdm bar in the parent stays clean.
    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    try:
        mesh = get_mesh()  # → enthält Torch Tensoren
        if mesh is None:
            queue.put(None)
            return

        tmpfile = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_mesh_{idx}.pt"
        )
        torch.save(mesh, tmpfile.name)
        queue.put(tmpfile.name)

    except Exception as e:
        print(f"Fehler im Worker: {e}")
        queue.put(None)


# Hilfsfunktion: führt get_mesh in eigenem Prozess aus, killt bei Timeout
def run_with_timeout(idx, timeout=300, quiet=False):
    q = mp.Queue()
    p = mp.Process(target=_mesh_worker, args=(q, idx, quiet))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # Timeout → kein Mesh

    return q.get() if not q.empty() else None


def main(quiet=True, number_of_meshes=1000):
    checkpoint_interval = 10  # alle 10 speichern
    checkpoint_dir = "./saved_meshes/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    successful_meshes = 0
    failed_meshes = 0
    counter = 0
    database = []  # sammelt Meshes bis 10

    # quiet -> tqdm bar (total trials + successful), no per-mesh prints.
    # verbose -> full prints (incl. gmsh), no bar.
    pbar = tqdm(total=number_of_meshes, desc="meshes", unit="mesh", disable=not quiet)

    while successful_meshes < number_of_meshes:
        tmp_path = run_with_timeout(counter, timeout=300, quiet=quiet)
        counter += 1
        if not quiet:
            print(f"\n--- counter: {counter} ---\n")

        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                # Mesh im Hauptprozess laden
                mesh_data = torch.load(tmp_path, weights_only=False)
                os.remove(tmp_path)  # Temp-Datei wieder löschen

                successful_meshes += 1
                database.append(mesh_data)
                pbar.update(1)
                if not quiet:
                    print(f"successful meshes: {successful_meshes}")

                # Alle 10 abspeichern
                if successful_meshes % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f'checkpoint_mesh_{
                            successful_meshes}.pt'
                    )
                    torch.save(database, checkpoint_path)
                    database = []  # RAM freigeben
                    if not quiet:
                        print(f"Checkpoint gespeichert: {checkpoint_path}")

            except Exception as e:
                failed_meshes += 1
                if not quiet:
                    print(f"⚠️ Fehler beim Laden des Meshes: {e}")

        else:
            failed_meshes += 1
            if not quiet:
                print("⚠️ Mesh ist nicht valide oder Timeout erreicht")

        # tqdm postfix: total trials, fails, live success rate
        pbar.set_postfix(trials=counter, fails=failed_meshes,
                         rate=f"{successful_meshes / max(counter, 1):.1%}")

    pbar.close()

    # letzten Rest speichern (<10)
    if database:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt'
        )
        torch.save(database, checkpoint_path)
        print(f"Final checkpoint gespeichert: {checkpoint_path}")

    print(f"Total failed meshes {
          failed_meshes}; total successful meshes {successful_meshes}")


def extract_mesh_data(tri_mesh, quad_mesh, block_mesh):

    # Extract features of the triangulated mesh incl. frame_field & streamline generation

    tri_coordinates = tri_mesh.x
    tri_faces = tri_mesh.faces
    tri_edges = tri_mesh.edge_index
    tri_edges_attr = tri_mesh.edge_attr
    tri_mesh_face_attr = tri_mesh.face_attr
    streamlines = tri_mesh.streamlines
    frame_field_angle = tri_mesh.frame_field_iteration_number
    frame_field_u = tri_mesh.u
    singularities_coords = tri_mesh.singularities_coords
    frame_field_iteration_number = tri_mesh.frame_field_iteration_number

    frame_field_time = tri_mesh.time_frame_field_generator
    singularities = tri_mesh.singularities

    # Extract the block structure
    blocking_nodes = block_mesh.x
    blocking_faces = block_mesh.faces
    edge_to_streamline = block_mesh.edge_to_streamline
    # Extract transfinite Interpolated mesh
    quad_coordinates = quad_mesh.x
    quad_faces = quad_mesh.faces
    quad_edges = quad_mesh.edge_index

    final_mesh = Data(blocking_nodes=blocking_nodes,
                      blocking_faces=blocking_faces,
                      quad_coordinates=quad_coordinates,
                      quad_faces=quad_faces, quad_edges=quad_edges,
                      singularities=singularities,
                      singularities_coords=singularities_coords,
                      frame_field_time=frame_field_time,
                      frame_field_iteration_number=frame_field_iteration_number,
                      frame_field_u=frame_field_u,
                      frame_field_angle=frame_field_angle,
                      streamlines=streamlines,
                      edge_to_streamline=edge_to_streamline,
                      tri_edges_attr=tri_edges_attr,
                      tri_mesh_face_attr=tri_mesh_face_attr,
                      tri_edges=tri_edges, tri_faces=tri_faces,
                      tri_coordinates=tri_coordinates)

    return final_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quad-domain-partition data generator")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Full output (incl. gmsh), no progress bar. Default: silent + tqdm bar.")
    parser.add_argument("-n", "--number", type=int, default=1000,
                        help="Number of successful meshes to generate (default: 1000)")
    args = parser.parse_args()
    main(quiet=not args.verbose, number_of_meshes=args.number)
