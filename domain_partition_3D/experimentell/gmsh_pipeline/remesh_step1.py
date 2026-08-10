#!/usr/bin/env python3
"""
T1_9 Hub/Shroud Quasi-Structured Quad Remeshing (Step 1)

Usage:
    python3 remesh_step1.py --surface hub --elem-size 1.0
    python3 remesh_step1.py --surface shroud --elem-size 1.0
    python3 remesh_step1.py --both --elem-size 1.0

Parameters:
    --elem-size: Target element size (default: 1.0)
                 Smaller = more elements, finer mesh
                 Larger = fewer elements, coarser mesh
    
    --surface-angle: Angle for surface classification in radians (default: pi/2)
                     Controls how surfaces are split/parametrized
    
    --curve-angle: Angle for curve merging in radians (default: pi/2)
                   CRITICAL: Must be pi/2 to merge boundary curves
                   If too small, creates 200+ curves → infinite loop
    
    --timeout: Subprocess timeout in seconds (default: 60)
    
    --output-dir: Output directory (default: /root/repos/block_structured_meshing)

Key findings:
    - classifySurfaces with curveAngle=pi/2 merges the 204 boundary edges into 2 curves
    - Mesh.QuadqsSizemapMethod=0 forces the background field to be used
    - Algorithm 11 = QuadQuasiStructured (Packing of Parallelograms)
    - Must use subprocess with timeout to avoid gmsh infinite loops
"""

import argparse
import os
import subprocess
import sys

# Default paths
DEFAULT_OUT_DIR = "/root/repos/block_structured_meshing"
DEFAULT_STL_DIR = "/root/repos/block_structured_meshing"

# Python interpreter with gmsh installed
PYTHON = "/root/venv/bin/python3"


def run_gmsh_remesh(stl_file, out_msh, elem_size, surface_angle, curve_angle, timeout):
    """
    Run gmsh remeshing in a subprocess with timeout.
    """
    # Build the gmsh script
    script = f"""
import gmsh
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 11)
    
    # Load STL
    gmsh.merge("{stl_file}")
    print("[INFO] Loaded STL: {stl_file}")
    
    # Classify surfaces
    # CRITICAL: curveAngle must be large enough to merge boundary curves
    # Default pi/2 merges the 204 boundary edges into 2 curves
    gmsh.model.mesh.classifySurfaces(
        {surface_angle},   # surface angle
        True,              # includeBoundary
        True,              # forceParametrizablePatches
        {curve_angle}      # curveAngle (CRITICAL!)
    )
    print("[INFO] Classified surfaces")
    
    # Check entities
    n_curves = len(gmsh.model.getEntities(1))
    n_surfaces = len(gmsh.model.getEntities(2))
    print(f"[INFO] Entities: {{n_curves}} curves, {{n_surfaces}} surfaces")
    
    # Create geometry from mesh
    gmsh.model.mesh.createGeometry()
    print("[INFO] Created geometry")
    
    # Background field for element size control
    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field, "F", "{elem_size}")
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    print(f"[INFO] Set background field: size={elem_size}")
    
    # Mesh options
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", {elem_size})
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", {elem_size})
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.QuadqsSizemapMethod", 0)
    
    # Generate 2D mesh
    print("[INFO] Generating 2D mesh...")
    gmsh.model.mesh.generate(2)
    
    # Get mesh info
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    n_quads = 0
    n_tris = 0
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 3:   # 4-node quad
            n_quads = len(etags)
        elif etype == 2: # 3-node triangle
            n_tris = len(etags)
    
    print(f"[RESULT] {{n_quads}} quads, {{n_tris}} tris, {{n_nodes}} nodes")
    
    # Save both formats
    gmsh.write("{out_msh}")
    print(f"[SAVED] {out_msh}")
    
    out_vtk = "{out_msh}".replace(".msh", ".vtk")
    gmsh.write(out_vtk)
    print(f"[SAVED] {out_vtk}")
    
    gmsh.finalize()
    
except Exception as e:
    print(f"[ERROR] {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Write temporary script
    tmp_script = f"{os.path.dirname(out_msh)}/tmp_remesh_{os.path.basename(out_msh)}.py"
    with open(tmp_script, 'w') as f:
        f.write(script)
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/root/venv/lib/python3.12/site-packages'
    
    # Run with timeout
    try:
        result = subprocess.run(
            [PYTHON, tmp_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"[ERROR] Process exited with code {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] TIMEOUT after {timeout}s")
        return False
    finally:
        # Cleanup
        if os.path.exists(tmp_script):
            os.remove(tmp_script)


def main():
    parser = argparse.ArgumentParser(
        description="T1_9 Hub/Shroud Quad Remeshing - Step 1"
    )
    parser.add_argument(
        "--surface", 
        choices=["hub", "shroud", "both"],
        default="both",
        help="Which surface to remesh"
    )
    parser.add_argument(
        "--elem-size",
        type=float,
        default=1.0,
        help="Target element size (default: 1.0). Smaller = finer mesh."
    )
    parser.add_argument(
        "--surface-angle",
        type=float,
        default=3.14159/2,
        help="Surface classification angle in radians (default: pi/2)"
    )
    parser.add_argument(
        "--curve-angle",
        type=float,
        default=3.14159/2,
        help="Curve merging angle in radians (default: pi/2). CRITICAL: must be large enough to merge boundary curves!"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Subprocess timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Determine surfaces to process
    surfaces = []
    if args.surface in ["hub", "both"]:
        surfaces.append(("hub", f"{DEFAULT_STL_DIR}/T1_9_hub_raw.stl"))
    if args.surface in ["shroud", "both"]:
        surfaces.append(("shroud", f"{DEFAULT_STL_DIR}/T1_9_shroud_raw.stl"))
    
    # Process each surface
    for name, stl_file in surfaces:
        out_file = f"{args.output_dir}/output/T1_9/{name}/{name}_quad.msh"
        
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"STL: {stl_file}")
        print(f"Output: {out_file}")
        print(f"Element size: {args.elem_size}")
        print(f"Surface angle: {args.surface_angle:.4f} rad")
        print(f"Curve angle: {args.curve_angle:.4f} rad")
        print(f"Timeout: {args.timeout}s")
        print(f"{'='*60}\n")
        
        success = run_gmsh_remesh(
            stl_file=stl_file,
            out_msh=out_file,
            elem_size=args.elem_size,
            surface_angle=args.surface_angle,
            curve_angle=args.curve_angle,
            timeout=args.timeout
        )
        
        if success:
            print(f"\n[SUCCESS] {name} remeshing complete")
        else:
            print(f"\n[FAILED] {name} remeshing failed")
    
    print("\n" + "="*60)
    print("Step 1 complete. Next: extract separatrices and detect blocks")
    print("="*60)


if __name__ == "__main__":
    main()
