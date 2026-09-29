"""Pyxa to SpatialData: read the xsmall crop with the pyxa reader and browse it by z-plane.

    uv run marimo edit demo_pyxa.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import fsspec
    import marimo as mo
    import matplotlib.pyplot as plt
    from spatialdata import get_extent

    from spatialdata_io.experimental import pyxa

    DATA_DIR = Path(__file__).resolve().parent / "data"
    return DATA_DIR, Path, fsspec, get_extent, mo, plt, pyxa


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pyxa → SpatialData demo

    Reads the `xsmall` crop (a 100 × 100 × 100 µm cube, 187 cells) of the
    [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) dataset with the experimental
    `pyxa` reader from [spatialdata-io](https://github.com/scverse/spatialdata-io), writes it to SpatialData
    Zarr, and browses it one z-plane at a time.

    ## 1. Get the data

    Downloads `xsmall/` from the Hugging Face Hub through fsspec's `hf://` filesystem (provided by
    `huggingface_hub`), skipping files already in `data/xsmall/`; the reader reads the zipped OME-Zarr DAPI
    mosaic in place.
    """)
    return


@app.cell
def _(DATA_DIR, Path, fsspec):
    xsmall_dir = DATA_DIR / "xsmall"
    xsmall_dir.mkdir(parents=True, exist_ok=True)

    fs = fsspec.filesystem("hf")
    for remote_path in fs.ls("datasets/Stellaromics/demo/xsmall", detail=False):
        local_path = xsmall_dir / Path(remote_path).name
        if not local_path.exists():
            fs.get(remote_path, str(local_path))

    sorted(p.name for p in xsmall_dir.iterdir())
    return (xsmall_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Convert to SpatialData

    The reader returns every element in µm in the `global` coordinate system. Segmentation polygons are
    stored on disk in pixels, one per cell per z-plane; the reader converts and repairs them into
    `cell_boundaries_z` (per plane, with the plane centre in a `Z_um` column) and `cell_boundaries` (one
    footprint per cell, annotated by the `rna` table). The result is written to SpatialData Zarr.
    """)
    return


@app.cell
def _(DATA_DIR, pyxa, xsmall_dir):
    sdata = pyxa(xsmall_dir)
    sdata.write(DATA_DIR / "xsmall.zarr", overwrite=True)
    sdata
    return (sdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Browse z-planes

    The full-resolution DAPI stack is loaded into memory, and transcripts and z-plane polygons are grouped by the
    image plane their z falls in, so each slider step only looks up one plane. Transcripts are blue if assigned to
    a cell and orange if not.
    """)
    return


@app.cell
def _(get_extent, sdata):
    image = sdata["mosaic_image"]["scale0"]["image"]
    stack = image.isel(c=0).values  # (z, y, x) DAPI planes, in memory
    extent = get_extent(image)
    z_min, z_max = extent["z"]
    dz = (z_max - z_min) / len(stack)


    def plane_of(z):
        return ((z - z_min) // dz).astype(int)


    transcripts = sdata["transcripts"].compute()
    polygons = sdata["cell_boundaries_z"]
    transcripts_by_plane = dict(list(transcripts.groupby(plane_of(transcripts["z"]))))
    polygons_by_plane = dict(list(polygons.groupby(plane_of(polygons["Z_um"]))))
    return dz, extent, polygons_by_plane, stack, transcripts, transcripts_by_plane, z_min


@app.cell
def _(mo, stack):
    plane = mo.ui.slider(0, len(stack) - 1, value=len(stack) // 2, label="z-plane", show_value=True)
    plane
    return (plane,)


@app.cell
def _(dz, extent, plane, plt, polygons_by_plane, stack, transcripts, transcripts_by_plane, z_min):
    k = plane.value
    tx = transcripts_by_plane.get(k, transcripts.iloc[:0])
    assigned = tx["assigned"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(stack[k], cmap="gray", extent=(*extent["x"], extent["y"][1], extent["y"][0]))
    if k in polygons_by_plane:
        polygons_by_plane[k].boundary.plot(ax=ax, color="white", linewidth=0.6)
    ax.scatter(tx.loc[assigned, "x"], tx.loc[assigned, "y"], s=8, c="#2a78d6", label="assigned")
    ax.scatter(tx.loc[~assigned, "x"], tx.loc[~assigned, "y"], s=8, c="#eb6834", label="unassigned")
    ax.set(
        xlim=extent["x"],
        ylim=(extent["y"][1], extent["y"][0]),
        xlabel="x (µm)",
        ylabel="y (µm)",
        title=f"z = {z_min + (k + 0.5) * dz:.2f} µm (plane {k}): "
        f"{len(polygons_by_plane.get(k, []))} polygons, {len(tx)} transcripts",
    )
    ax.legend(loc="upper right")
    fig
    return


if __name__ == "__main__":
    app.run()
