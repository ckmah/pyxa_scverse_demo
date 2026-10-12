# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.0",
#     "milume==1.3.4",
#     "scanpy",
#     "igraph",
#     # pyxa reader: ckmah/spatialdata-io, pyxa-reader branch, pinned to its head commit
#     "spatialdata-io @ git+https://github.com/ckmah/spatialdata-io@9a85304fc2f11f04e187f161c399dfb3a7900b3c",
#     "spatialdata",
#     "fsspec",
#     "huggingface-hub",
#     "polars",
# ]
# ///
"""Pyxa to SpatialData: read the small crop with the pyxa reader and explore it in Milume.

    uv run marimo edit 2_pyxa_spatialdata.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **molab:** Switch from **Preview** to **Server** (*Run on server*). The widget needs a live kernel to download data and render; Preview alone will not run this notebook.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import fsspec
    import marimo as mo
    import polars as pl

    from spatialdata_io.experimental import pyxa

    DATA_DIR = Path(__file__).resolve().parent / "data"
    return DATA_DIR, Path, fsspec, mo, pl, pyxa


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🔬 Pyxa 3D spatial omics → SpatialData

    Reads the `small` crop (a ~1,320 × 450 × 130 µm slab of mouse brain, 4,372 cells, ~290 MB) of the
    [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) dataset with the experimental
    `pyxa` reader from [spatialdata-io](https://github.com/scverse/spatialdata-io), writes it to SpatialData
    Zarr, and explores it in the Milume widget.

    ## 1. Download the data

    Downloads `small/` from the Hugging Face Hub through fsspec's `hf://` filesystem (provided by
    `huggingface_hub`).
    """)
    return


@app.cell
def _(DATA_DIR, Path, fsspec):
    small_dir = DATA_DIR / "small"
    small_dir.mkdir(parents=True, exist_ok=True)

    fs = fsspec.filesystem("hf")
    for remote_path in fs.ls(
        "datasets/Stellaromics/demo/small", detail=False
    ):
        local_path = small_dir / Path(remote_path).name
        if not local_path.exists():
            fs.get(remote_path, str(local_path))

    sorted(p.name for p in small_dir.iterdir())
    return (small_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    ## 2. Convert to SpatialData

    The reader returns every element in µm in the `global` coordinate system, then writes SpatialData Zarr.
    /// admonition | Tip\n
    The `pyxa()` reader will read the zipped OME-Zarr DAPI mosaic in place. No need to unzip.
    ///
    """),
            mo.mermaid("""
    %%{init: {"theme": "base", "themeVariables": {"lineColor": "#888888", "textColor": "#888888", "edgeLabelBackground": "transparent", "clusterBkg": "transparent", "clusterBorder": "#888888"}, "flowchart": {"nodeSpacing": 25, "rankSpacing": 90, "curve": "basis"}}}%%
    flowchart LR
        subgraph IN["Input files"]
            direction TB
            A1(mosaic_3d.ome.zarr.zip)
            A2(cell_assigned_gene_v1.csv)
            A3(segmentation_geometries_v1.parquet)
            A5(cell_metadata_v1.csv)
            A4(cell_by_gene_v1.csv)
        end
        subgraph OUT["SpatialData"]
            direction TB
            B1(images/mosaic_image)
            B2(points/transcripts)
            B3(shapes/cell_boundaries_z)
            B4(shapes/cell_boundaries)
            B5(tables/rna)
        end
        A1 --> B1
        A2 --> B2
        A3 --> B3
        A3 --> B4
        A5 --> B5
        A4 --> B5
        classDef file fill:#6b7280,stroke:#9ca3af,color:#ffffff
        classDef img fill:#2563eb,stroke:#60a5fa,color:#ffffff
        classDef pts fill:#d97706,stroke:#fbbf24,color:#ffffff
        classDef shp fill:#16a34a,stroke:#4ade80,color:#ffffff
        classDef tbl fill:#db2777,stroke:#f472b6,color:#ffffff
        style IN fill:none,stroke:#888888,color:#888888
        style OUT fill:none,stroke:#888888,color:#888888
        class A1,A2,A3,A4,A5 file
        class B1 img
        class B2 pts
        class B3,B4 shp
        class B5 tbl
    """),
        ]
    )
    return


@app.cell
def _(pl, small_dir):
    pl.read_csv(small_dir / "cell_assigned_gene_v1.csv")
    return


@app.cell
def _(pl, small_dir):
    pl.read_csv(small_dir / "cell_by_gene_v1.csv")
    return


@app.cell
def _(small_dir):
    from spatialdata_io.readers.pyxa import (
        _get_image,
    )  # internal function for demo purposes only; use pyxa

    _get_image(small_dir / "mosaic_3d.ome.zarr.zip")
    return


@app.cell
def _(pl, small_dir):
    pl.read_csv(small_dir / "cell_metadata_v1.csv")
    return


@app.cell
def _(DATA_DIR, pyxa, small_dir):
    sdata = pyxa(small_dir, labels=True)
    sdata.write(DATA_DIR / "small.zarr", overwrite=True)
    sdata
    return (sdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Explore in the Milume widget

    `milume.peek` opens the SpatialData object in an interactive 3D viewer: pan and zoom the DAPI stack, draw
    landmarks, and open the Inspect view on any region. **Note:** `small` is a mouse brain subset used here
    only to walk through the tools; it says nothing about colon biology (see `3_colon_3d_milume.py` for that).
    """)
    return


@app.cell
def _(sdata):
    import scanpy as sc

    _adata = sdata.tables["rna"]
    sc.pp.log1p(_adata)
    sc.pp.pca(_adata, random_state=0)
    sc.pp.neighbors(_adata, random_state=0)
    sc.tl.umap(_adata, random_state=0)
    sc.tl.leiden(_adata, flavor="igraph", n_iterations=2, resolution=0.8, random_state=0)
    _adata.obs["leiden"] = _adata.obs["leiden"].astype("category")
    clustered = sdata
    return (clustered,)


@app.cell(expand_output=True)
def _(clustered, mo):
    import milume

    peek = mo.ui.anywidget(milume.peek(clustered, color="leiden"))
    peek
    return


if __name__ == "__main__":
    app.run()
