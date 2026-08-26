try:
    from cli import main
except ImportError:
    from domain_partition.cli import main
import os


def _default_paths():
    data_dir = os.environ.get("QUADMESH_DATA_DIR", "./data")
    input_path = os.environ.get(
        "QUADMESH_PARTITION_INPUT",
        os.path.join(data_dir, "data_domain_partition_v2.pt"),
    )
    output_path = os.environ.get(
        "QUADMESH_PARTITION_OUTPUT",
        os.path.join(data_dir, "structured_quad_meshes.pt"),
    )
    return input_path, output_path


if __name__ == "__main__":
    import argparse

    input_default, output_default = _default_paths()
    parser = argparse.ArgumentParser(
        description="Transfinite structured quad mesh generator"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=input_default,
        help=f"input .pt file (default: {input_default})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=output_default,
        help=f"output .pt file (default: {output_default})",
    )
    parser.add_argument(
        "-d",
        "--divisions",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="transfinite divisions (default: 3 4 5)",
    )
    args = parser.parse_args()
    main(
        [
            "--input",
            args.input,
            "--output",
            args.output,
            "--divisions",
            *[str(d) for d in args.divisions],
            "--strict",
        ]
    )
