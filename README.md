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
Run01 / Analysis02 / A2 (local Pyxa output, not public). The notebook is one
widget, `LandmarksWidget(sdata)`: from the SpatialData it finds the cell table,
the 3D cell labels it annotates and the 3D nuclear image on the same grid.
Cells are colored by Pyxa Studio `Cluster`. **Inspect** (`I`, then click the
tissue) opens a floating cube of that 500 µm window over the full ~142 µm
stack; its controls (tilt, additive / maximum-intensity rendering, **Labels**,
X / Y / Z cuts, contrast) sit in the map's toolbar. A table counts the clusters
inside the cube: the inspect window within the cut box (`landmarks.volume_cut`).

It needs spatial-rx with the Landmarks inspect cube (not released yet),
installed editable from a sibling checkout (`../spatial-rx`), with its widget
bundles built, and the `pyxa` reader with optional inputs from the sibling
`spatialdata-io` checkout.

1. **Build** `data/colon_a2.sdata.zarr` (about 13 GB, 12 GB of it the mosaic):
   - `tables/rna` from the `pyxa` reader: counts and cell metadata (required)
     plus `pyxa_studio_v1.csv` for `Cluster` and the 3D UMAP, keeping the cells
     Pyxa Studio kept, in Pyxa µm. Transcripts are skipped (7.7 GB).
   - `labels/cell_labels`: the 23M per-plane segmentation polygons rasterized
     onto the 3D mosaic's level-0 voxel grid as a `Labels3DModel` (`uint32`,
     label = the `N` of `Region_N`), with the mosaic's transform and pyramid.
     The table annotates it through `label_id`.
   - `images/mosaic`: Meteor's `Region/mosaic/mosaic_3d.ome.zarr` (DAPI,
     `uint8`) copied level by level as an `Image3DModel` with the labels'
     transform, in `(1, 32, 256, 256)` chunks (about 90 s).

   `--no-mosaic` skips the image, `--no-labels` the labels (table alone in
   about 30 s), and `--only-mosaic` adds the image to an existing build
   without re-rasterizing the labels.

   ```bash
   uv run python build_colon_a2.py --overwrite --source "/path/to/A2/Analysis Group"
   ```

2. **Open** the notebook:

   ```bash
   uv run marimo edit colon_a2.py
   ```

Landmarks packs all 1,020 genes for the gene picker (about 1.4 GB for 358k
cells, with a warning), so the map takes about 45 s to appear. The cube reads only the
256×256×32 chunks its window covers.
