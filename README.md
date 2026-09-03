# quadmesh

AI-accelerated block-structured mesh generation for multi-physics
turbomachinery optimization.

This package bundles three research code bases that together form a
single pipeline from a geometric domain description to a
block-structured all-quad mesh suitable for CFD and eigenfrequency
analysis:

- `domain_partition/`    -- 2D GNN-based cross/frame field generation
- `domain_partition_3D/` -- 3D data generation for turbomachinery surfaces
- `meshtron/`            -- Transformer-based quad mesh tokenization

The package is a research deliverable of DFG SPP 2353 "Daring More
Intelligence" and the primary contribution of the associated PhD thesis


## Data pipeline

```
                     +--------------------------+
                     |  geometric domain input  |
                     |  (STL / MSH / boundaries)|
                     +-----------+--------------+
                                 |
            +--------------------+--------------------+
            |                                         |
            v                                         v
   +-------------------+                   +-----------------------+
   |  domain_partition |                   |  domain_partition_3D  |
   |  (2D GNN frame    |                   |  (3D data generation,  |
   |   field, CSME 25) |                   |   hub/shroud, seam)    |
   +---------+---------+                   +-----------+-----------+
             |                                         |
             |  2D frame field +                       |  3D surface blocks
             |  singularity graph                     |  + TFI / Xiao
             v                                         v
                  +--------------------------------------+
                  |     meshtron (Transformer)          |
                  |  tokenize -> hourglass transformer  |
                  |  -> detoken -> quad mesh            |
                  |  (AIFLUIDS 26 tokenization)   |
                  +-----------------+--------------------+
                                    |
                                    v
                  +--------------------------------------+
                  |  block-structured all-quad mesh      |
                  |  -> CFD + eigenfrequency evaluation  |
                  +--------------------------------------+
```

End-to-end, the pipeline takes a turbomachinery geometry, produces a
2D frame field and a 3D surface decomposition, learns a tokenized
representation of the resulting quad layout, and reconstructs an
all-quad mesh that downstream solvers can consume without remeshing.

## Installation

The package is shipped as a single installable Python package. All
three subdirectories are exposed under the `quadmesh` import namespace.

```bash
# editable install from the repository root
pip install -e /root/repos/duty/quadmesh
```

`pyproject.toml` declares the runtime dependencies:

- `torch`
- `numpy`
- `gmsh`
- `trimesh`
- `networkx`

Python >= 3.10 is required. For GPU training of `meshtron`, install a
CUDA-enabled PyTorch that matches the local driver; the rest of the
stack is CPU-only.

## Repository layout

```
quadmesh/
|-- domain_partition/        # 2D GNN cross/frame field (CSME 2025)
|   |-- main.py              # CLI entry: train / evaluate frame field
|   |-- data_generator.py    # synthetic 2D shape / mesh synthesis
|   |-- quadmesh_generator.py
|   |-- cli.py               # CLI entry: env-var driven config
|   |-- fieldgen/            # learned frame-field model + losses
|   `-- tests/
|
|-- domain_partition_3D/     # 3D data generation for turbomachinery
|   |-- domain_partition.py  # CLI: hub / shroud, ta / tb / xiao
|   |-- dp3d/                # pipeline package
|   |   |-- field/           # vendored 2D cross-field tools
|   |   |-- extraction.py
|   |   |-- unwrap_surface.py
|   |   |-- partition_surface.py
|   |   |-- tmesh.py
|   |   `-- xiao.py
|   |-- data/                # T1_9 test case
|   `-- experimentell/       # Gmsh pipeline, 3D extrapolation
|
|-- meshtron/                # Transformer quad-mesh tokenization
|   |-- main.py              # CLI entry
|   |-- meshtron.py          # hourglass-transformer model
|   |-- hourglass_transformer.py
|   |-- attention.py         # attention variants
|   |-- embedding.py
|   |-- point_encoder.py
|   |-- faceCount_encoder.py
|   |-- dataset.py
|   |-- detoken.py           # token -> quad mesh decoder
|   |-- half_edge.py         # half-edge mesh utilities
|   `-- plotting_tools.py
|
|-- quadmesh/                # Python package glue
|   `-- __init__.py
|
|-- pyproject.toml           # PEP 621 build config
`-- README.md
```

The three research code bases are kept as subdirectories with their
own CLI entry points so that each can be developed and cited
independently. The top-level `quadmesh/` package only re-exports the
common utilities for shared imports.

All three are **git submodules**, not plain directories:

| Path | Upstream | Visibility |
|------|----------|------------|
| `domain_partition/`    | `RentschlerTobias/domain_partition` (HTTPS) | public |
| `domain_partition_3D/` | `RentschlerTobias/domain_partition_3D` (SSH) | private |
| `meshtron/`            | `RentschlerTobias/quadtron` (SSH) | private |

Clone with `--recurse-submodules`, or run `git submodule update --init
--recursive` in an existing checkout. The two private submodules use SSH
URLs and need a key on the GitHub account; `domain_partition` is public
and clones anonymously over HTTPS.

## Usage

Each subdirectory is runnable on its own; the importable package is
intended for downstream consumers (optimizer, MCP server, etc.).

```bash
# 2D GNN frame-field training
python domain_partition/main.py --config configs/2d_frame_field.yaml

# 3D block partition on a turbomachinery volume mesh
python domain_partition_3D/domain_partition.py \
    domain_partition_3D/data/T1_9/T1_9_ru_gridGmsh.msh \
    --part both --method ta --plots

# Transformer-based quad mesh tokenization
python meshtron/main.py --mode train --config configs/meshtron.yaml
```

## References

The methods implemented here were introduced and validated in the
following publications:

- **CSME 2025** -- 2D GNN cross/frame-field generation. Graph neural
  network predicting a smooth cross field on 2D planar domains with
  explicit singularity handling; published in the proceedings of the
  CSME conference 2025.
- **AIFLUIDS 2026** -- Memory efficient tokenization for quad block-structured 
  meshes. accepted at AIFLUIDS 2026 

  ## Project context

- DFG Priority Programme SPP 2353 "Daring More Intelligence",
  work packages WP4 (online optimization) and WP5 (3D meshing).
- PhD thesis, Institute of Fluid Mechanics and Hydraulic Machinery 
  (IHS), Universitat Stuttgart.
- The `optimizer/` repository (separate) consumes the meshes produced
  here for coupled CFD + eigenfrequency optimization via RL and DE.

## Roadmap

- **2D  Quad block-structured Tokenization** (AIFLUIDS 2026) -- Memory efficient tokenization 
  for transformer of block-structured quad meshes on uniform flowchannel around a random naca profil.
- ** Domain partition of 3D quad surfaces (3rd order quad meshes) ** (Aug–Dez 2026)
  -- Extend  data generation pipline from 2d uniform flow channel with random naca profil 
  to 3D spline surfaces (3rd order quadrilateral elements) block structures on hub/shroud surfaces.
  Quad-spline patch representation for smooth higher-order surface
  decomposition of cylindrical turbomachinery geometries.
- **3D Higher-Order Spline-Surfaces Transformer* 
  Transformer to gernerate Quad-spline patch 

## Attribution

`domain_partition`, `domain_partition_3D`, `presentations_privat` and
`conference_papers` were migrated from the University of Stuttgart TIK
GitLab to GitHub with their full commit history. Commit metadata was
normalised in the process. This section records what changed and why, so
the rewrite is auditable after the fact.

**Linked addresses.** Contributions are attributed to
`tobias.rentschler@ihs.uni-stuttgart.de` and `tobias-rentschler@gmx.de`.
Both must be *verified* on the GitHub account — GitHub only counts a
commit toward the contribution graph if its author address is verified
and linked. Addresses that are not linked produce commits that display
correctly but count for nobody.

**Identity normalisation.** Historic commits carried machine-local
identities from the institute cluster. These were rewritten with
`git filter-repo` (metadata only — file contents and tree hashes are
unchanged in `domain_partition_3D`, `presentations_privat` and
`conference_papers`):

| Old | New | Reason |
|-----|-----|--------|
| `tobis.rentschler@ihs.uni-stuttgart.de` | `tobias.rentschler@ihs.uni-stuttgart.de` | typo in the local git config |
| `trentschler@{mars,reynolds,snickers,vellamo}.ihs.uni-stuttgart.de` | `tobias.rentschler@ihs.uni-stuttgart.de` | default `user@host` from cluster nodes |
| `root@srv1715448.hstgr.cloud` | `tobias-rentschler@gmx.de` | commits made as root on a rented box |
| author name `trentschler`, `root` | `Tobias Rentschler` | internal TIK username / system account |

`ac136362@uni-stuttgart.de` (in `conference_papers`) was deliberately left
untouched, as were all ~380 third-party contributor identities in the
vendored reveal.js fork inside `presentations_privat`.

**Default-branch caveat.** Only commits reachable from a repository's
*default branch* appear in the contribution graph. Work that lives solely
on side branches — `hermes`, `perf_opti`, `quadMesh`, `airfoil`,
`framefield`, `manim_animation` in `domain_partition` — will never show
up there, regardless of authorship. The branches are kept because they
carry history, not for graph purposes.

**Retroactive display.** GitHub dates contributions by commit date, not
push date. The migrated commits therefore appear on their original days
going back to the start of the TIK history, not on the migration date.

**Known residue.** The same pre-normalisation addresses still exist in
repositories that were already public before this migration — `quadmesh`
itself, `eigenfrequencies`, and `fun/*`. Rewriting those would mean
force-pushing published history, so it was deliberately not done. Two
notebooks that used to live in `domain_partition/` and are still in this
repository's history (`StreamlinePostProcessing.ipynb`, `Untitled.ipynb`)
contain cluster paths in their stored cell outputs; they were removed
from the working tree with the submodule migration but remain reachable
in older `quadmesh` commits.

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
