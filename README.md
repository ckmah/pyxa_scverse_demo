# pyxa_scverse_demo

[marimo](https://marimo.io) notebooks showing how to read Stellaromics Pyxa
output into a [SpatialData](https://spatialdata.scverse.org/) object with the
experimental `pyxa` reader from
[spatialdata-io](https://github.com/scverse/spatialdata-io), browse it one
z-plane at a time (`demo_pyxa.py`), and analyse a full region with scverse
(`colon_a2.py`).

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

Each notebook also declares its dependencies inline (PEP 723 `# /// script`,
with `demo_pyxa.py`'s reader pinned to a commit of the `pyxa-reader` branch of
`ckmah/spatialdata-io`),
so it runs in a marimo sandbox or on [molab](https://molab.marimo.io) without the
sibling checkout:

```bash
uvx marimo edit --sandbox demo_pyxa.py
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

## Spatial analysis of colorectal cancer with 3D context (`colon_a2.py`)

Glasgow colon H1K, Run01 / Analysis02 / A2: the `colon/` folder of
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo)
(attribution on the dataset card).

1. **Build** `data/colon_a2.sdata.zarr` (~13 GB; ~8 min, ~30 GB peak memory). `--download`
   first fetches the five Hub files the build needs into `data/hf/colon/`: **22.5 GB**
   (12.0 GB mosaic, 9.3 GB segmentation polygons, 1.15 GB counts + metadata + Pyxa Studio
   export; the 7.7 GB of transcripts stay on the Hub).

   ```bash
   uv run python build_colon_a2.py --download --overwrite
   ```

2. **Open** the notebook: the Milume widget (`milume.peek`), a guide to its tools, then
   two vignettes that read the landmarks you draw: **shape → composition by depth** and
   **line + wide buffer → expression gradients**.

   ```bash
   uv run marimo edit colon_a2.py
   ```

`colon_a2_common.py` holds the helpers: loading the store and joining the cell-type,
lineage and Novae-domain annotations from `annotations/colon_a2/`, picking the marker
genes the widget packs, the toolbar icons for the tools guide, and the two vignettes'
measures and plots.

`tests/smoke_colon_a2.py` runs the notebook headlessly on synthetic data built from the
annotations (no store needed), with and without drawn landmarks; `--sdata` points it at
a real store.
