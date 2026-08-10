
from tools import MeshGenerator, NACA_airfoil 
import numpy as np
import torch
import os
import multiprocessing as mp
import time

def get_mesh():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())
    
    try:
        airfoil                     = NACA_airfoil()
        random_lc                   =  0.04+ 0.02*np.random.rand()
        mesh_gen                    = MeshGenerator(airfoil, quadMesh=True, lc=random_lc)
        return mesh_gen.mesh
    except Exception as e:
        print(f'\n domain partition failed: {e} \n')
 
def main():

    number_of_meshes =10000 
    checkpoint_interval = 1000  # Speichere alle x erfolgreiche Meshes
    checkpoint_dir = "./saved_meshes/quad_meshes/checkpoints_quad_meshes"

    os.makedirs(checkpoint_dir, exist_ok=True)

    database = []
    successful_meshes = 0  
    failed_meshes = 0  
    counter = 0
    for n in range(number_of_meshes):
        is_valid = False
        
        while is_valid == False:
            try:
                mesh_data = mp.Pool(1).apply_async(get_mesh).get(timeout=300)

            except Exception as e:
                      print(f"timeout_error: {e}")
            counter += 1             
            print(f"\n \n \n counter: {counter} \n \n \n")               
            if mesh_data is not None:
                is_valid = True    
                database.append(mesh_data)
                successful_meshes += 1
                print(f"successful meshes: {successful_meshes}")
                if (successful_meshes)%checkpoint_interval==0:
                    try:
                            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
                            torch.save(database, checkpoint_path)
                            database = [] #resett database
                            print(f"Checkpoint gespeichert: {checkpoint_path}")
                    except Exception as checkpoint_error:
                            print(f"Warnung: Fehler beim Speichern des Checkpoints: {checkpoint_error}")
            else:
                failed_meshes += 1
                print(f"Warning: Transifinite Mesh is not valid")

    print(f'total failed meshes {failed_meshes}; total successful meshes {successful_meshes }') 

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
    torch.save(database, checkpoint_path)
    print(f"Final Checkpoint reached")

if __name__ == "__main__":                                                                         
    main()
