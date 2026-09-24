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
