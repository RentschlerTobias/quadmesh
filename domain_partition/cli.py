import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from tqdm import tqdm


def _load_tool(name):
    """Load a domain_partition/tools submodule directly, avoiding tools/__init__.py."""
    tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    spec = importlib.util.spec_from_file_location(
        f"domain_partition.tools.{name}", os.path.join(tools_dir, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QuadPartitionValidator = _load_tool("quad_partition_validator").QuadPartitionValidator


def _default_data_dir():
    return os.environ.get("QUADMESH_DATA_DIR", "./data")


def _load_meshes(path):
    item = torch.load(path, weights_only=False)
    if isinstance(item, tuple) and len(item) == 2:
        return [item]
    if isinstance(item, list):
        return item
    return [item]


def _as_tri_mesh(mesh):
    return Data(
        x=getattr(mesh, "tri_coordinates", mesh.x),
        faces=getattr(mesh, "tri_faces", mesh.faces),
        edge_index=getattr(mesh, "tri_edges", mesh.edge_index),
        edge_attr=getattr(mesh, "tri_edges_attr", mesh.edge_attr),
        frame_field=getattr(mesh, "frame_field_u", None),
    )


def _as_blocked_mesh(mesh):
    return Data(
        x=getattr(mesh, "blocking_nodes", mesh.x),
        faces=getattr(mesh, "blocking_faces", mesh.faces),
        edge_index=getattr(mesh, "blocking_edge_index", mesh.edge_index),
    )


def _unpack_sample(mesh):
    """Return (blocked_mesh, tri_mesh) either from a tuple or from a Data object."""
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        return mesh[0], mesh[1]
    return _as_blocked_mesh(mesh), _as_tri_mesh(mesh)


def transfinite_process(input_path, output_path, divisions, strict=True):
    QuadMeshGenerator = _load_tool("transfinite_quadMesh_generator").QuadMeshGenerator
    meshes = _load_meshes(input_path)
    data = []
    for mesh in tqdm(meshes, desc="mesh process"):
        blocked, tri = _unpack_sample(mesh)
        for division in divisions:
            generator = QuadMeshGenerator(blocked, transfinite_divisions=division)
            mesh_transfinite = generator.transfinite_mesh
            if strict:
                validator = QuadPartitionValidator(
                    mesh_transfinite, tri, frame_field=tri.frame_field, strict=True
                )
                if not validator.is_valid():
                    print(f"strict gate rejected division {division}")
                    for d in validator.diagnostics():
                        print(f"  {d}")
                    continue
            mesh_transfinite.tri_coordinates = tri.x
            data.append(mesh_transfinite)
    torch.save(data, output_path)
    print(f"saved {len(data)} meshes to {output_path}")


def validate_meshes(input_path, strict=True):
    meshes = _load_meshes(input_path)
    all_valid = True
    for idx, mesh in enumerate(meshes):
        blocked, tri = _unpack_sample(mesh)
        validator = QuadPartitionValidator(
            blocked, tri, frame_field=tri.frame_field, strict=strict
        )
        valid = validator.is_valid()
        print(f"mesh {idx}: valid={valid}")
        if not valid:
            all_valid = False
            for d in validator.diagnostics():
                print(f"  {d}")
    return all_valid


def main(argv=None):
    parser = argparse.ArgumentParser(description="2D quad domain partition CLI")
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="input .pt file (list of meshes or single mesh)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output .pt file",
    )
    parser.add_argument(
        "-d",
        "--divisions",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="transfinite divisions (default: 3 4 5)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run strict validation gate (default: True)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate input only; no output file written",
    )
    args = parser.parse_args(argv)

    if args.validate:
        if not args.input:
            parser.error("--validate requires --input")
        ok = validate_meshes(args.input, strict=args.strict)
        return 0 if ok else 1

    if not args.input or not args.output:
        parser.error("--input and --output required (or --validate --input)")

    out_dir = Path(args.output).parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    transfinite_process(args.input, args.output, args.divisions, strict=args.strict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
