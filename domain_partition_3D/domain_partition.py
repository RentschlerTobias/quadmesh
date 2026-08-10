#!/usr/bin/env python3
"""Block-structured quad domain partition of cylindrical hub/shroud surfaces.

Pipeline: MSH/STL input -> hub/shroud extraction (MSH only) -> cylinder
unwrap 3D->2D -> cross-field guided quad partition (methods: ta, tb, xiao)
-> TFI grid fill (ta/tb). Optional: boundary-layer quad-block extraction
from the hex core (--include-boundary) and step-by-step showcase plots
(--plots).

Examples:
    python domain_partition.py data/T1_9/T1_9_ru_gridGmsh.msh --part hub
    python domain_partition.py data/T1_9/T1_9_ru_gridGmsh.msh \\
        --part both --method ta tb xiao --plots --include-boundary
    python domain_partition.py data/T1_9/T1_9_hub_raw.stl --method ta --plots
"""

import argparse
import sys
from pathlib import Path

METHODS = ("ta", "tb", "xiao")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="volume mesh (.msh, tagged hub/shroud) or an "
                         "already extracted single surface (.stl)")
    ap.add_argument("--part", choices=("hub", "shroud", "both"),
                    default="hub",
                    help="which surface(s) to process (default: hub; for "
                         ".stl input this only names the output directory)")
    ap.add_argument("--method", nargs="+", default=["ta"],
                    choices=METHODS + ("all",),
                    help="partition method(s) to run (default: ta)")
    ap.add_argument("--include-boundary", action="store_true",
                    help="also extract the block-structured boundary-layer "
                         "quad meshes from the hex core (.msh input only)")
    ap.add_argument("--plots", action="store_true",
                    help="write analysis plots and the step01..step09 "
                         "showcase walkthrough")
    ap.add_argument("--output", type=Path, default=Path("output"),
                    help="output root directory (default: output/)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    methods = list(METHODS) if "all" in args.method else \
        list(dict.fromkeys(args.method))
    parts = ("hub", "shroud") if args.part == "both" else (args.part,)
    suffix = args.input.suffix.lower()

    from dp3d import extraction, plotting, tmesh, xiao
    from dp3d import clean_separatrix as cs
    from dp3d import unwrap_surface as us

    # module state of the validated T1 runs
    us.set_blade_tip_corners(False)
    cs.set_emanate_outer_corners(True)

    if suffix == ".msh":
        parsed = extraction.parse_msh(args.input)
        stls = extraction.extract_hub_shroud(
            args.input, args.output / "extracted", parts, parsed=parsed)
        if args.include_boundary:
            extraction.extract_boundary_quads(
                args.input, args.output / "extracted", parts, parsed=parsed)
    elif suffix == ".stl":
        if args.include_boundary:
            sys.exit("--include-boundary needs a .msh volume mesh")
        if args.part == "both":
            sys.exit("a .stl input is a single surface -- "
                     "use --part hub or --part shroud")
        stls = {args.part: args.input}
    else:
        sys.exit(f"unsupported input type: {suffix} (expected .msh or .stl)")

    for part, stl in stls.items():
        out_dir = args.output / part
        print(f"\n=== {part}: {stl} -> {out_dir} ===")
        runs = {}
        for method in methods:
            if method == "xiao":
                runs[method] = xiao.run_xiao(stl)
                xiao.write_xiao_report(runs[method], out_dir / "xiao")
            else:
                runs[method] = tmesh.run_tmesh(
                    stl, out_dir, tag=method,
                    continue_seam_edges=(method == "tb"))
        if args.plots:
            plotting.write_plots(stl, part, runs, args.output / "plots")


if __name__ == "__main__":
    main()
