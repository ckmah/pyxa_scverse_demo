# pyxa_scverse_demo

Jupyter notebook showing how to read Stellaromics Pyxa output into a
[SpatialData](https://spatialdata.scverse.org/) object with the experimental
`pyxa` reader from [spatialdata-io](https://github.com/scverse/spatialdata-io),
and browse it one z-plane at a time.

It uses the `xsmall` crop (100 × 100 × 100 µm) of the
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) dataset.

## Setup

The reader lives on the `pyxa-reader` branch of spatialdata-io, pending
upstream review, so until it is pushed the project installs it editable from a
sibling checkout:

```
spatialdata-io/        # pyxa-reader branch
pyxa_scverse_demo/     # this project
```

Then install and open the notebook with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab demo_pyxa.ipynb
```

## Notebook

1. **Get the data**: downloads `xsmall/` from the Hub via fsspec's `hf://`
   filesystem into `data/` and extracts the zipped OME-Zarr DAPI mosaic.
2. **Convert**: reads it with `pyxa(...)` and writes SpatialData Zarr to
   `data/xsmall.zarr`.
3. **Browse z-planes**: a slider over DAPI z-planes, with the cell polygons of
   that plane and the transcripts within half a plane of it drawn on top.

## Colon A2: clusters → 3D nuclei (`colon_a2.py`)

A [marimo](https://marimo.io) notebook on a full Region: Glasgow colon H1K,
Run01 / Analysis02 / A2 (local Pyxa output, not public). Cells are colored by
Pyxa Studio `Cluster` in a spatial-rx `LandmarksWidget`. **Inspect** on the
tissue picks a 500 µm window; a `VolumeCubeWidget`, floating in a wigglystuff
`FloatingPanel`, loads it from Meteor's 3D DAPI mosaic over the full ~142 µm
stack. The cube's own controls tilt it from top-down to side-on, switch
additive / maximum-intensity rendering, show segmented cells (**Labels**) and
cut it in X, Y and Z. Focusing a cluster in the map's legend, or a Selection,
fills those cells in the cube in their map colours. A table breaks down the
clusters inside the cut.

It needs spatial-rx with windowed cube loading (not released yet), installed editable from a sibling checkout
(`../spatial-rx`), with its widget bundles built, and the `pyxa` reader with
optional inputs from the sibling `spatialdata-io` checkout.

1. **Build** `data/colon_a2.sdata.zarr` (about 1.8 GB, 4 min on 60 workers):
   - `tables/rna` from the `pyxa` reader: counts and cell metadata (required)
     plus `pyxa_studio_v1.csv` for `Cluster` and the 3D UMAP, keeping the cells
     Pyxa Studio kept, in Pyxa µm. Transcripts are skipped (7.7 GB).
   - `labels/cell_labels`: the 23M per-plane segmentation polygons rasterized
     onto the 3D mosaic's level-0 voxel grid as a `Labels3DModel` (`uint32`,
     label = the `N` of `Region_N`), with the mosaic's transform and pyramid.
     The table annotates it through `label_id`.

   The mosaic is not copied: the notebook reads
   `Region/mosaic/mosaic_3d.ome.zarr` in place, which shares the same grid.
   `--no-labels` builds the table alone in about 30 s.

   ```bash
   uv run python build_colon_a2.py --overwrite --source "/path/to/A2/Analysis Group"
   ```

2. **Open** the notebook:

   ```bash
   uv run marimo edit colon_a2.py
   ```

Landmarks packs all 1,020 genes for the gene picker (about 216 MB for 358k
cells), so the map takes about 45 s to appear. Mosaics written by Meteor with
256x256 store chunks read about 34 MB per inspect window; older
1024x1024-chunk mosaics still work but read about 145 MB.
