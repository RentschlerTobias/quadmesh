
from tools import MeshGenerator, NACA_airfoil
import numpy as np
# in console the command: ../fieldgen/fieldgen mesh.obj mesh_out.obj --degree=4 --alignToBoundary --s=0


airfoil = NACA_airfoil()
random_lc =0.02+ 0.08*np.random.rand()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
mesh_gen.export_to_obj(filename="mesh.obj")


def main():

    airfoil = NACA_airfoil()
    random_lc =0.02+ 0.08*np.random.rand()
    mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
    mesh_gen.export_to_obj_field(filename="mesh.obj")


if __name__ == "__main__":
    main()
