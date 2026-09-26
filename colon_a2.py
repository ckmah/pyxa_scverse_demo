"""Glasgow colon A2: Pyxa clusters on Landmarks, the nuclei in the inspect cube.

Loads the SpatialData built by ``build_colon_a2.py`` into ``data/`` (not committed):
the full-section cell table with Pyxa Studio cluster labels, in Pyxa µm, the 3D
cell labels and Meteor's 3D nuclear mosaic on one voxel grid. ``LandmarksWidget(sdata)``
finds all three; **Inspect** (``I``) opens a cube of the mosaic, sized to the
square (its size follows zoom).

    uv run python build_colon_a2.py --overwrite
    uv run marimo edit colon_a2.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import spatialdata as sd

    from spatial_rx import LandmarksWidget

    SDATA_PATH = Path(__file__).resolve().parent / "data" / "colon_a2.sdata.zarr"
    CLUSTER = "Cluster"
    return CLUSTER, LandmarksWidget, SDATA_PATH, mo, pd, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Colon A2: clusters on the map, nuclei in 3D

    Press **I** (Inspect) and click the tissue: a floating cube loads that
    window of Meteor's 3D nuclear stain over the full stack depth, with the
    segmented cells — the square (its size follows zoom). Its controls sit in
    the map's toolbar. The table counts the clusters inside the cube (its
    window, cut by the X / Y / Z cuts).
    """)
    return


@app.cell(expand_output=True)
def _(CLUSTER, LandmarksWidget, SDATA_PATH, mo, sd):
    mo.stop(
        not SDATA_PATH.exists(),
        mo.md(
            f"`{SDATA_PATH}` not found. Build it first:\n\n"
            "```bash\nuv run python build_colon_a2.py --overwrite\n```"
        ),
    )
    sdata = sd.read_zarr(SDATA_PATH)
    widget = LandmarksWidget(sdata, color=CLUSTER, contrast_limits=(40, 255))
    landmarks = mo.ui.anywidget(widget)
    landmarks
    return landmarks, sdata


@app.cell
def _(CLUSTER, landmarks, mo, pd, sdata):
    # The cube's cut box [x0, x1, y0, y1, z0, z1] in µm, written by the widget
    # when a cut slider is released or a new window settles. It starts at the
    # whole volume; the cube shows it within the inspect window.
    cut = list(landmarks.volume_cut)
    mo.stop(len(cut) != 6, mo.md("No cube: this SpatialData has no 3D image."))
    mo.stop(
        landmarks.inspect_cx is None,
        mo.md("### In the cube\n\nPress **I** and click the tissue to load a window."),
    )
    half = landmarks.inspect_size_um / 2
    for axis, center in ((0, landmarks.inspect_cx), (1, landmarks.inspect_cy)):
        cut[2 * axis] = max(cut[2 * axis], center - half)
        cut[2 * axis + 1] = min(cut[2 * axis + 1], center + half)
    adata = sdata["rna"]
    xyz = adata.obsm["spatial"]
    mask = True
    for axis in range(3):
        mask = mask & (xyz[:, axis] >= cut[2 * axis]) & (xyz[:, axis] <= cut[2 * axis + 1])
    counts = adata.obs.loc[mask, CLUSTER].value_counts()
    extent = " · ".join(f"{ax} {cut[2 * i]:.0f}–{cut[2 * i + 1]:.0f}" for i, ax in enumerate("XYZ"))
    mo.vstack(
        [
            mo.md(f"### In the cube\n\n**{int(mask.sum()):,}** cells in {extent} µm"),
            mo.ui.table(
                pd.DataFrame({"cells": counts[counts > 0]}).rename_axis("cell type").reset_index(),
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
