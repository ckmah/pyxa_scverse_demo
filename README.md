# pyxa_scverse_demo

[marimo](https://marimo.io) notebooks showing how to read Stellaromics Pyxa
output into a [SpatialData](https://spatialdata.scverse.org/) object with the
experimental `pyxa` reader from
[spatialdata-io](https://github.com/scverse/spatialdata-io), browse it one
z-plane at a time (`demo_pyxa.py`), and analyse a full region with scverse
(`colon_a2.py` and beat notebooks).

## Setup

The reader lives on the `pyxa-reader` branch of spatialdata-io, pending
upstream review, so until it is pushed the project installs it editable from a
sibling checkout:

```
spatialdata-io/        # pyxa-reader branch
pyxa_scverse_demo/     # this project
```

Then install and open a notebook with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run marimo edit demo_pyxa.py
```

## Pyxa to SpatialData (`demo_pyxa.py`)

A marimo notebook on the `xsmall` crop (100 × 100 × 100 µm) of the
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) dataset.

1. **Get the data**: downloads `xsmall/` from the Hub via fsspec's `hf://`
   filesystem into `data/xsmall/`, skipping files already there; the reader reads the zipped OME-Zarr DAPI mosaic in
   place.
2. **Convert**: reads it with `pyxa(...)` and writes SpatialData Zarr to
   `data/xsmall.zarr`.
3. **Browse z-planes**: a slider over DAPI z-planes, with the cell polygons of
   that plane and the transcripts within half a plane of it drawn on top.

## Colon A2: why 3D matters

A [marimo](https://marimo.io) demo on a full Region: Glasgow colon H1K,
Run01 / Analysis02 / A2, published as the `colon/` folder of
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) (raw Pyxa
output, ~30 GB; attribution on the dataset card). The narrative contrasts what a
**flat 2D map implies** with what **3D inspection and measurement** show, for a
mixed scverse audience (tool builders, cell biologists, pathologists, biomedical
researchers, platform folks).

### Layout

- **`colon_a2_common.py`** — shared SpatialData load, annotation joins
  (`cell_typing.parquet`, `novae_domains.parquet`, niche names), and marker-gene
  picker for `LandmarksWidget(genes=...)`.
- **`colon_a2.py`** — thin index linking the five beat notebooks.
- **Beat notebooks** (one runnable moment each):

| Beat | Notebook | Flatten → collapse |
|------|----------|-------------------|
| 1 — Rings aren't rings | `colon_a2_beat1_rings.py` | Crypt **rings** → **tubes through Z** (Inspect + orbit) |
| 2 — Cut like a pathologist | `colon_a2_beat2_cross_section.py` | Single-plane **arcs** → **continuous walls** (Cross-section) |
| 3 — Neighbors lie in 2D | `colon_a2_beat3_neighbors.py` | Flat-map **neighbors** → microns apart in **Z** (`nearest_distances`) |
| 4 — Stacked niches | `colon_a2_beat4_niches.py` | Flat composition mix → **composition by depth** (1 µm z bins) |
| 5 — See → analyze | `colon_a2_beat5_analyze.py` | Viewing → **pick cells** → `in_selection` + `rank_genes_groups` DE |

It needs [Milume](https://github.com/ckmah/milume) 1.1.0 or later (from PyPI, formerly
spatial-rx) for the Landmarks inspect cube, `nearest_distances`, and `z_bin_size`
measures, plus the `pyxa` reader with optional inputs from the sibling
`spatialdata-io` checkout.

1. **Download** the region with `--download` (below): the counts, cell
   metadata, Pyxa Studio export, segmentation polygons and zipped mosaic from
   `colon/` into `data/hf/colon/` (about 22 GB; the 7.7 GB of transcripts stay
   on the Hub). `--source` also takes a Pyxa `Analysis Group` directory instead.

2. **Build** `data/colon_a2.sdata.zarr` (about 13 GB) with one `pyxa(..., labels=True)`
   call:
   - `tables/rna`: counts and cell metadata (required) plus `pyxa_studio_v1.csv`
     for `Cluster` and the 3D UMAP; this script keeps only the cells Pyxa
     Studio kept, in Pyxa µm. Transcripts are skipped (7.7 GB). `X` holds the
     raw integer counts (sparse `int64`; the previous build wrote `float32`).
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

   Takes about 8 minutes total (roughly 1.5 min to read, 6 min to write),
   with a peak of roughly 30 GB of memory.

3. **Open** the index or a beat notebook:

   ```bash
   uv run marimo edit colon_a2.py
   uv run marimo edit colon_a2_beat1_rings.py
   ```

The beat notebooks pass ~40 marker genes to `LandmarksWidget(genes=...)`, so the
widget packs about 56 MB and each notebook runs in about 10 s; all 1,020
genes would be about 1.4 GB and 45 s. The cube reads only the 256×256×32 chunks
its window covers.
