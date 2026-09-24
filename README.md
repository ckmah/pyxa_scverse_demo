# pyxa_scverse_demo

Demo notebook showing how to read Pyxa (Stellaromics/Meteor-APA pipeline)
analysis-group output into a [SpatialData](https://spatialdata.scverse.org/)
object, using the experimental `pyxa` reader from
[spatialdata-io](https://github.com/scverse/spatialdata-io) (currently on a
[fork branch](https://github.com/ckmah/spatialdata-io/tree/pyxa-reader)
pending upstream review).

The notebook streams a full (non-subsampled) demo dataset from
[Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) on
the Hugging Face Hub via the `hf://` fsspec filesystem, builds the
`SpatialData` object, and plots the transcripts and segmentation shapes.

## Running

The notebook is a [marimo](https://marimo.io) notebook with inline
dependencies (PEP 723), runnable directly with [uv](https://docs.astral.sh/uv/):

```bash
uv run demo_pyxa_hf.py
```

This opens the notebook in your browser; no separate environment setup is
needed.
