
from tools import MeshGenerator, FrameField, NACA_airfoil
import torch
import os
import numpy as np

def main():
    meshes = []
    base_dir = "./saved_meshes/frame_field_time_measured_v2"
    os.makedirs(base_dir, exist_ok=True)

    num_meshes_to_generate = 100
    save_points = 10

    for i in range(num_meshes_to_generate):
        airfoil = NACA_airfoil()
        random_lc =0.02+ 0.08*np.random.rand()
        mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
        frameField = FrameField(mesh_gen.mesh)

        meshes.append(frameField.mesh)

        if i % save_points == 0:
            filename = os.path.join(base_dir, f"num_meshes_{i}.pt")
            torch.save(meshes, filename)

    # Save final version
    final_filename = os.path.join(base_dir, "all_meshes.pt")
    torch.save(meshes, final_filename)

if __name__ == "__main__":
    main()
