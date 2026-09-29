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
   filesystem into `data/`; the reader reads the zipped OME-Zarr DAPI mosaic in
   place.
2. **Convert**: reads it with `pyxa(...)` and writes SpatialData Zarr to
   `data/xsmall.zarr`.
3. **Browse z-planes**: a slider over DAPI z-planes, with the cell polygons of
   that plane and the transcripts within half a plane of it drawn on top.

## Colon A2: scverse analysis, steered by landmarks (`colon_a2.py`)

A [marimo](https://marimo.io) notebook on a full Region: Glasgow colon H1K,
Run01 / Analysis02 / A2, published as the `colon/` folder of
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) (raw Pyxa
output, ~30 GB; attribution on the dataset card). The analysis is plain
scverse; `LandmarksWidget(sdata)` adds landmarks drawn on the tissue as
per-cell coordinates for scanpy.

1. **Markers**: scanpy normalizes the counts (kept in `layers["counts"]`),
   ranks markers per Pyxa Studio `Cluster` and names each cluster by its top
   marker (`obs["cluster"]`); dotplot and the Pyxa UMAP.
2. **Landmarks**: from the SpatialData the widget finds the cell table, the 3D
   cell labels and the 3D nuclear image. Cells are colored by the named
   clusters, with the markers (41 genes) in the gene picker. **Inspect** (`I`,
   then click the tissue) opens a floating cube of a 300 µm window over the
   full stack; **Save** keeps it as a selection.
3. **Measure**: `spatial_rx` `composition`, `distances` and `along_positions`
   on the picked landmark, computed in XY with every cell's depth in 1 µm z
   bins. Composition per z bin, cell-type profiles vs distance or along a path
   (with their depth), and gene profiles as scanpy matrixplots over the binned
   `obs` columns (`dist_<landmark>`, `path_s`, `z_bin`).
4. **Selection vs rest**: `obs["in_selection"]` from a widget selection, then
   `sc.tl.rank_genes_groups` against every other cell.
5. **In the cube**: cluster counts inside the inspect window within the cut
   box (`landmarks.volume_cut`).

It needs spatial-rx with the Landmarks inspect cube and `z_bin_size` measures
(on `main`, not released yet),
installed editable from a sibling checkout (`../spatial-rx`), with its widget
bundles built, and the `pyxa` reader with optional inputs from the sibling
`spatialdata-io` checkout.

1. **Download** the region with `--download` (below): the counts, cell
   metadata, Pyxa Studio export, segmentation polygons and zipped mosaic from
   `colon/` into `data/hf/colon/` (about 22 GB; the 7.7 GB of transcripts stay
   on the Hub). `--source` also takes a Pyxa `Analysis Group` directory instead.

2. **Build** `data/colon_a2.sdata.zarr` (about 13 GB) with one `pyxa(..., labels=True)`
   call:
   - `tables/rna`: counts and cell metadata (required) plus `pyxa_studio_v1.csv`
     for `Cluster` and the 3D UMAP; this script keeps only the cells Pyxa
     Studio kept, in Pyxa µm. Transcripts are skipped (7.7 GB).
   - `labels/cell_labels`: the segmentation polygons rasterized onto the 3D
     mosaic's grid as a `Labels3DModel` (`uint32`, label = the `N` of
     `Region_N`), all pyramid levels. The table annotates it through
     `label_id`.
   - `images/mosaic_image`: Meteor's `mosaic_3d.ome.zarr` (DAPI, `uint8`),
     read from its zip in place, all pyramid levels, on the same grid as the
     labels.

   ```bash
   uv run python build_colon_a2.py --download --overwrite
   ```

   Takes about 7 minutes total (roughly 1.5 min to read, 5.5 min to write),
   with a peak of roughly 44 GB of memory.

3. **Open** the notebook:

   ```bash
   uv run marimo edit colon_a2.py
   ```

The notebook passes the 41 marker genes to `LandmarksWidget(genes=...)`, so the
widget packs about 56 MB and the whole notebook runs in about 10 s; all 1,020
genes would be about 1.4 GB and 45 s. The cube reads only the 256×256×32 chunks
its window covers.
