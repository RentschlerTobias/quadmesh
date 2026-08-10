#!/usr/bin/env python3
"""Run Shroud T-mesh partition with Hub-Master prescribed singularities.

Usage:
    python run_shroud_tmesh.py [tag]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dp3d.tmesh import run_tmesh


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ta"
    root = Path(__file__).resolve().parent.parent.parent

    # Load prescribed singularities from Hub-Master projection
    prescribed_path = root / "output" / "T1_9" / "shroud_stage1" / f"shroud_prescribed_{tag}.json"
    if not prescribed_path.exists():
        print(f"ERROR: {prescribed_path} not found. Run project_singularities.py first.")
        sys.exit(1)

    print(f"[shroud] loading prescribed singularities from {prescribed_path}")
    with open(prescribed_path) as f:
        shroud_data = json.load(f)

    prescribed = shroud_data["singularities"]
    print(f"[shroud] {len(prescribed)} prescribed singularities loaded")

    stl = str(root / "T1_9_shroud_raw.stl")
    out = root / "output" / "T1_9" / "shroud_stage1" / "tmesh"

    result = run_tmesh(
        stl=stl,
        out_dir=out,
        tag=f"shroud_{tag}",
        prescribed_singularities=prescribed,
    )

    print(f"[shroud] partition complete: {result['metrics']['blocks']} blocks, "
          f"{result['metrics']['rejected_regions']} rejected")


if __name__ == "__main__":
    main()
