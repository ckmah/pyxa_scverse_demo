# pyxa_scverse_demo

marimo notebooks for the Stellaromics workshop on 3D spatial omics with Pyxa, SpatialData and Milume.

| Notebook | What it covers | Run |
| --- | --- | --- |
| [`1_intro_marimo.py`](1_intro_marimo.py) | Intro to marimo: reactive cells, UI elements and markdown | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/stellaromics/pyxa_scverse_demo/blob/c5f9efdd460e929e2a6b9d5b7943dcbd737833ef/1_intro_marimo.py) |
| [`2_pyxa_spatialdata.py`](2_pyxa_spatialdata.py) | Read Pyxa output into SpatialData and explore it in the Milume widget | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/stellaromics/pyxa_scverse_demo/blob/c5f9efdd460e929e2a6b9d5b7943dcbd737833ef/2_pyxa_spatialdata.py) |
| [`3_colon_3d_milume.py`](3_colon_3d_milume.py) | Analysis of 3D spatial omics in colorectal cancer with Milume | Run on your own machine; needs ~30 GB of memory and a ~22 GB download |

## Run the colon notebook locally

```
uv run python build_colon_a2.py --download --overwrite
uv run marimo edit 3_colon_3d_milume.py
```

The build downloads ~22.5 GB from the Hub, writes a ~13 GB SpatialData store to `data/`, and peaks at ~30 GB of memory.

Data: [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) on Hugging Face.
