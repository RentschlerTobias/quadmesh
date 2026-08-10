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
"KI-beschleunigte Blockstruktur-Generierung fuer multiphysikalische
Optimierung von Turbomaschinen".

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
                  |  (AIFLUIDS 26 polar tokenization)   |
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
|   |-- fieldgen/            # learned frame-field model + losses
|   |-- jupyter_notebooks/   # exploratory notebooks
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
- **AIFLUIDS 2026** -- Polar tokenization for quad meshes. Polar
  coordinate tokenization that respects rotational symmetry on
  turbomachinery boundaries; accepted at AIFLUIDS 2026 (Springer /
  Open Access proceedings).
- **Springer SPP-Buch 2027** -- 3D quad-spline block decomposition of
  cylindrical turbomachinery surfaces. Extended treatment of the 3D
  hub/shroud pipeline with quad-spline patches; planned contribution
  to the DFG SPP 2353 Springer book (2027).

## Project context

- DFG Priority Programme SPP 2353 "Daring More Intelligence",
  work packages WP4 (online optimization) and WP5 (3D meshing).
- PhD thesis, Institut fuer Luft- und Raumfahrtantriebe und -systeme
  (ILR), Universitat Stuttgart.
- The `optimizer/` repository (separate) consumes the meshes produced
  here for coupled CFD + eigenfrequency optimization via RL and DE.

## License

Internal research code. Distribution and reuse are governed by the
rules of DFG SPP 2353 and the contributing institutions.