"""Beat 4: stacked niches — composition by depth."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from colon_a2_common import CELL_TYPE, Z_BIN, apply_mpl_theme, load_colon_a2
    from milume import LandmarksWidget, composition, landmarks_to_geodataframe

    return (
        CELL_TYPE,
        LandmarksWidget,
        Z_BIN,
        apply_mpl_theme,
        composition,
        landmarks_to_geodataframe,
        load_colon_a2,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Beat 4 — Stacked niches

    **Flatten:** a single composition bar **collapses depth** into one mix.

    **Collapse:** draw a shape (or a buffered line) over a region and run **Composition by depth**:
    cell-type, lineage, or domain proportions in 1 µm z bins show **stacked niches** — what a
    flat summary hides.

    [← Back to index](colon_a2.py)
    """)
    return


@app.cell
def _(load_colon_a2):
    sdata, adata, groups, marker_genes = load_colon_a2()
    return adata, groups, marker_genes, sdata


@app.cell(expand_output=True)
def _(CELL_TYPE, LandmarksWidget, marker_genes, mo, sdata):
    widget = LandmarksWidget(
        sdata, color=CELL_TYPE, genes=marker_genes, contrast_limits=(40, 255)
    )
    landmarks = mo.ui.anywidget(widget)
    landmarks
    return (landmarks,)


@app.cell
def _(mo):
    get_landmark, set_landmark = mo.state(None)
    get_selection, set_selection = mo.state("all")
    return get_landmark, get_selection, set_landmark, set_selection


@app.cell
def _(get_landmark, get_selection, landmarks, mo, set_landmark, set_selection):
    landmark_ids = [str(lm["id"]) for lm in landmarks.landmarks if not lm.get("hidden")]
    selection_ids = ["all"] + [str(s["id"]) for s in landmarks.selections]
    landmark_pick = mo.ui.dropdown(
        options=landmark_ids or ["(none)"],
        value=get_landmark() if get_landmark() in landmark_ids else (landmark_ids or ["(none)"])[-1],
        label="Landmark",
        on_change=set_landmark,
    )
    selection_pick = mo.ui.dropdown(
        options=selection_ids,
        value=get_selection() if get_selection() in selection_ids else "all",
        label="Cells",
        on_change=set_selection,
    )
    return landmark_pick, selection_pick


@app.cell
def _(groups, mo):
    group_pick = mo.ui.dropdown(options=groups, value="cell type", label="Group by")
    return (group_pick,)


@app.cell
def _(Z_BIN, adata, composition, group_pick, landmark_pick, landmarks, landmarks_to_geodataframe, mo, np, pd, plt, selection_pick):
    MIN_CELLS = 20

    def heatmap(ax, table, *, y_width=None, label=""):
        v = np.asarray(table.index, dtype=float)
        edges = np.append(v, v[-1] + y_width)
        mesh = ax.pcolormesh(
            np.arange(len(table.columns) + 1),
            edges,
            table.to_numpy(dtype=float),
            cmap="magma",
        )
        ax.set_xticks(np.arange(len(table.columns)) + 0.5, table.columns, rotation=90)
        ax.figure.colorbar(mesh, ax=ax, label=label, shrink=0.8)

    gdf = landmarks_to_geodataframe(
        [lm for lm in landmarks.landmarks if str(lm["id"]) == landmark_pick.value]
    )
    if len(gdf) == 0:
        result = mo.md("_Draw a shape (or buffered line) landmark on the map._")
    else:
        lid = str(gdf["id"].iloc[0])
        key = group_pick.value
        cells = landmarks.get_obs_names(adata, selection_id=selection_pick.value)
        comp = composition(adata, gdf, obs_key=key, obs_names=cells, z_bin_size=Z_BIN)
        if comp.empty:
            result = mo.md(
                "_**Composition by depth** needs a shape, or a line / spline with a buffer, that covers the cells._"
            )
        else:
            by_z = comp.pivot_table(index="z_bin", columns="group", values="proportion", fill_value=0)
            n_per_z = comp.groupby("z_bin")["n_total"].first()
            by_z = by_z.where(n_per_z.reindex(by_z.index) >= MIN_CELLS)
            z_edges = np.arange(comp["z_bin"].min(), comp["z_bin"].max() + 1.5 * Z_BIN, Z_BIN)
            by_z = by_z.reindex(
                index=z_edges[:-1],
                columns=[g for g in adata.obs[key].cat.categories if g in by_z.columns],
            )
            pooled = comp.groupby("group")["count"].sum().reindex(by_z.columns)
            fig, (ax_bar, ax_z) = plt.subplots(
                1, 2, figsize=(11, 5), width_ratios=[1, 2.2], layout="constrained"
            )
            ax_bar.barh(pooled.index[::-1], (pooled / pooled.sum())[::-1])
            ax_bar.set(xlabel="proportion, whole depth (flat summary)", title=f"{int(pooled.sum()):,} cells")
            heatmap(ax_z, by_z, y_width=Z_BIN, label="proportion in z bin")
            ax_z.set(ylabel="z (µm)", title=f"Composition per {Z_BIN:g} µm z bin · {lid}")
            plt.close(fig)
            result = fig
    mo.vstack([mo.hstack([landmark_pick, selection_pick, group_pick], justify="start"), result], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
