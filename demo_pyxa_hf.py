# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "setuptools",
#     "spatialdata-io @ git+https://github.com/ckmah/spatialdata-io.git@pyxa-reader",
#     "huggingface_hub",
#     "fsspec==2026.9.0",
#     "spatialdata-plot==0.4.0",
#     "matplotlib==3.11.2",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Pyxa reader demo, streamed from Hugging Face

        Full (non-subsampled) analysis-group output from
        [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo)
        on the Hugging Face Hub, fetched via the `hf://` fsspec filesystem
        registered by `huggingface_hub`.
        """
    )
    return (mo,)


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import fsspec

    from spatialdata_io.experimental import pyxa

    return Path, fsspec, pyxa, tempfile


@app.cell
def _(mo):
    mo.md("""
    ## Download the dataset locally via `hf://` fsspec
    """)
    return


@app.cell
def _(Path, fsspec, tempfile):
    fs = fsspec.filesystem("hf")
    remote_dir = "datasets/Stellaromics/demo"

    local_dir = Path(tempfile.mkdtemp()) / "pyxa_demo_data"
    local_dir.mkdir(parents=True, exist_ok=True)

    files = [f["name"] for f in fs.ls(remote_dir) if f["name"].endswith((".csv", ".parquet"))]
    for remote_path in files:
        fs.get(remote_path, str(local_dir / Path(remote_path).name))

    sorted(p.name for p in local_dir.iterdir())
    return (local_dir,)


@app.cell
def _(mo):
    mo.md("""
    ## Build the SpatialData object
    """)
    return


@app.cell
def _(local_dir, pyxa):
    sdata = pyxa(local_dir)
    {
        "transcripts": len(sdata["transcripts"]),
        "cell_shapes": len(sdata["cell_shapes"]),
        "rna": sdata["rna"].shape,
    }
    return (sdata,)


@app.cell
def _(mo):
    mo.md("""
    ## Visualize each element
    """)
    return


@app.cell
def _(sdata):
    import spatialdata_plot  # noqa: F401
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sdata.pl.render_points(element="transcripts", color="assigned", size=1).pl.show(
        ax=axes[0], title="transcripts (points), colored by assigned"
    )
    sdata.pl.render_shapes(element="cell_shapes", color="ZIndex", cmap="viridis").pl.show(
        ax=axes[1], title="cell_shapes (segmentation polygons), colored by ZIndex"
    )
    fig.tight_layout()
    fig
    return


@app.cell
def _(sdata):
    sdata["rna"]
    return


if __name__ == "__main__":
    app.run()
