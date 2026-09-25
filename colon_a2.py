"""Glasgow colon A2: Pyxa clusters on Landmarks, the focused ones filled in the 3D nuclear cube.

Loads the SpatialData built by ``build_colon_a2.py`` into ``data/`` (not committed):
the full-section cell table with Pyxa Studio cluster labels, in Pyxa µm, and the
3D cell labels rasterized onto Meteor's ``mosaic_3d.ome.zarr``. The cube reads
that mosaic directly, loading only the 500 µm inspect window.

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
    import numpy as np
    import pandas as pd
    import spatialdata as sd
    from wigglystuff import FloatingPanel

    from spatial_rx import LandmarksWidget, VolumeCubeWidget

    SDATA_PATH = Path(__file__).resolve().parent / "data" / "colon_a2.sdata.zarr"
    # Inspect window (xy); the cube loads it over the full stack depth (~142 µm).
    WINDOW_UM = 500.0
    CLUSTER = "Cluster"
    return (
        CLUSTER,
        FloatingPanel,
        LandmarksWidget,
        Path,
        SDATA_PATH,
        VolumeCubeWidget,
        WINDOW_UM,
        mo,
        np,
        pd,
        sd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Colon A2: clusters on the map, nuclei in 3D

    Choose **Inspect** on the tissue map and click: the floating cube loads that
    500 µm window of Meteor's 3D nuclear stain. Focus a cluster in the map's
    legend, or a Selection, and the cube fills those cells in their map colours
    (**Labels** shows or hides them). The table counts the cells in the cube's cut.
    """)
    return


@app.cell
def _(CLUSTER, Path, SDATA_PATH, mo, sd):
    mo.stop(
        not SDATA_PATH.exists(),
        mo.md(
            f"`{SDATA_PATH}` not found. Build it first:\n\n"
            "```bash\nuv run python build_colon_a2.py\n```"
        ),
    )
    sdata = sd.read_zarr(SDATA_PATH)
    adata = sdata["rna"]
    mosaic_path = Path(str(adata.uns["pyxa"]["mosaic_3d"]))
    # Rasterized onto the mosaic's grid by build_colon_a2.py (absent with --no-labels).
    labels_path = SDATA_PATH / "labels" / "cell_labels" if "cell_labels" in sdata.labels else None
    cluster_names = [str(c) for c in adata.obs[CLUSTER].cat.categories]
    return adata, cluster_names, labels_path, mosaic_path


@app.cell
def _(CLUSTER, LandmarksWidget, WINDOW_UM, adata):
    landmarks_widget = LandmarksWidget(adata, color=CLUSTER)
    landmarks_widget.inspect_size_um = WINDOW_UM
    # One palette for the map, the cube and the charts.
    palette = landmarks_widget.category_colors(CLUSTER)
    return landmarks_widget, palette


@app.cell(expand_output=True)
def _(landmarks_widget, mo):
    landmarks = mo.ui.anywidget(landmarks_widget)
    landmarks
    return (landmarks,)


@app.cell
def _(VolumeCubeWidget, WINDOW_UM, labels_path, mosaic_path):
    cube = VolumeCubeWidget.from_ome_zarr(
        mosaic_path, labels_path=labels_path, window_size_um=WINDOW_UM, contrast_limits=(40, 255)
    )
    return (cube,)


@app.cell
def _(FloatingPanel, cube, mo):
    # The cube owns its controls (camera, projection, Labels, X/Y/Z cuts, contrast)
    # and syncs the cuts to Python on release. Defined here so it never remounts.
    cube_ui = mo.ui.anywidget(cube)
    FloatingPanel(
        cube_ui,
        corner="top-right",
        width=620,
        title=f"Nuclear cube · {cube.window_size_um:g} µm window",
    )
    return (cube_ui,)


@app.cell
def _(CLUSTER, adata, landmarks, landmarks_widget):
    # What the map's panel has focused: a cluster in the legend (indexed in legend
    # order), or a Selection.
    focus_kind, focus_index = landmarks.selected_kind, landmarks.selected_index
    legend = list(landmarks.legend_labels)
    in_selection = None
    if focus_kind == "type" and 0 <= focus_index < len(legend):
        in_selection = (adata.obs[CLUSTER].astype(str) == legend[focus_index]).to_numpy()
    elif focus_kind == "selection" and 0 <= focus_index < len(landmarks.selections):
        focused = landmarks.selections[focus_index]["id"]
        in_selection = adata.obs_names.isin(landmarks_widget.get_obs_names(adata, focused))
    return (in_selection,)


@app.cell
def _(CLUSTER, adata, cube, in_selection, landmarks, np, palette):
    if landmarks.inspect_cx is not None:
        cube.window_cx = float(landmarks.inspect_cx)
        cube.window_cy = float(landmarks.inspect_cy)
    # Fill the focused cells in the window, by cluster; only ids the cube can show.
    fill = np.zeros(adata.n_obs, dtype=bool) if in_selection is None else in_selection.copy()
    reach = cube.window_size_um / 2 + 10.0
    xy = adata.obsm["spatial"][:, :2]
    fill &= (np.abs(xy[:, 0] - cube.window_cx) <= reach) & (np.abs(xy[:, 1] - cube.window_cy) <= reach)
    filled = adata.obs.loc[fill, [CLUSTER, "label_id"]]
    cube.highlight_cells(
        {str(k): g["label_id"].astype(int).tolist() for k, g in filled.groupby(CLUSTER, observed=True)},
        palette,
    )
    return


@app.cell
def _(CLUSTER, adata, cube_ui, in_selection, landmarks, mo, pd):
    def shown(lo, hi, center, size):
        # The cube shows the window, cut by the X/Y slices.
        return max(lo, center - size / 2), min(hi, center + size / 2)

    if landmarks.inspect_cx is None:
        in_cut = mo.md(
            "### In the cube\n\nChoose **Inspect** on the map and click to load a window."
        )
    else:
        size = cube_ui.window_size_um
        cut_box = {
            "x": shown(cube_ui.slice_x_min, cube_ui.slice_x_max, cube_ui.window_cx, size),
            "y": shown(cube_ui.slice_y_min, cube_ui.slice_y_max, cube_ui.window_cy, size),
            "z": (cube_ui.slice_z_min, cube_ui.slice_z_max),
        }
        xyz = adata.obsm["spatial"]
        mask = True
        for axis, (lo, hi) in zip(range(3), cut_box.values()):
            mask = mask & (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
        everyone = adata.obs.loc[mask, CLUSTER].value_counts()
        table = pd.DataFrame({"cells": everyone})
        if in_selection is not None:
            table["focused"] = adata.obs.loc[mask & in_selection, CLUSTER].value_counts()
        table = table[table["cells"] > 0].fillna(0).astype(int).rename_axis("cell type").reset_index()
        extent = " · ".join(f"{ax.upper()} {lo:.0f}–{hi:.0f}" for ax, (lo, hi) in cut_box.items())
        in_cut = mo.vstack(
            [
                mo.md(f"### In the cube\n\n**{int(mask.sum()):,}** cells in the cut box {extent} µm"),
                mo.ui.table(table, selection=None, pagination=False),
            ]
        )
    in_cut
    return


if __name__ == "__main__":
    app.run()
