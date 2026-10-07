"""Beat 3: neighbors lie in 2D — nearest_distances XY vs 3D."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from colon_a2_common import CELL_TYPE, apply_mpl_theme, load_colon_a2
    from milume import LandmarksWidget, nearest_distances

    return (
        CELL_TYPE,
        LandmarksWidget,
        apply_mpl_theme,
        load_colon_a2,
        mo,
        nearest_distances,
        np,
        plt,
    )


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Beat 3 — Neighbors lie in 2D

    **Flatten:** lasso and neighborhood tools work on the **flat map** — a cell that looks like a
    neighbor in XY may sit **microns above or below** its nearest seed in Z.

    **Collapse:** pick a selection in **Cells** below (lasso, promoted neighborhood, or saved
    inspect cube). `milume.nearest_distances` pairs every cell with its nearest seed in XY and
    reports the depth offset `dz` — the gap a 2D view cannot see.

    [← Back to index](colon_a2.py)
    """)
    return


@app.cell
def _(load_colon_a2):
    sdata, adata, _groups, marker_genes = load_colon_a2()
    return adata, marker_genes, sdata


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
    get_selection, set_selection = mo.state("all")
    return get_selection, set_selection


@app.cell
def _(get_selection, landmarks, mo, set_selection):
    selection_ids = ["all"] + [str(s["id"]) for s in landmarks.selections]
    selection_pick = mo.ui.dropdown(
        options=selection_ids,
        value=get_selection() if get_selection() in selection_ids else "all",
        label="Cells",
        on_change=set_selection,
    )
    selection_pick
    return (selection_pick,)


@app.cell
def _(
    CELL_TYPE,
    adata,
    landmarks,
    mo,
    nearest_distances,
    np,
    plt,
    selection_pick,
):
    XY_RADIUS = 20.0
    Z_GAP = 10.0

    sel = selection_pick.value
    mo.stop(sel == "all", mo.md("_Pick a selection in **Cells** to use as seeds._"))
    seeds = list(landmarks.get_obs_names(adata, selection_id=sel))
    mo.stop(len(seeds) < 1, mo.md(f"_Selection **{sel}** is empty._"))
    neighbors = nearest_distances(adata, seeds, obs_key=CELL_TYPE)
    other = neighbors[~neighbors["seed"]]
    mo.stop(other.empty, mo.md("_No cells outside the seed set to compare._"))
    lying = other[(other["distance_xy"] < XY_RADIUS) & (other["dz"] > Z_GAP)]
    close_xy = other[other["distance_xy"] < XY_RADIUS]
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax_scatter.scatter(other["distance_xy"], other["dz"], s=1, alpha=0.15, color="0.6")
    if not lying.empty:
        ax_scatter.scatter(
            lying["distance_xy"],
            lying["dz"],
            s=6,
            alpha=0.6,
            label=f"XY < {XY_RADIUS:g} µm and |Δz| > {Z_GAP:g} µm (n={len(lying):,})",
        )
    ax_scatter.set(
        xlabel="distance to nearest seed in XY (µm)",
        ylabel="|Δz| to that seed (µm)",
        title="Flat-map neighbors vs depth offset",
    )
    if not lying.empty:
        ax_scatter.legend(loc="upper right", fontsize=10)
    if not close_xy.empty:
        ax_hist.hist(close_xy["dz"], bins=30, color="steelblue", edgecolor="none")
        med = float(np.median(close_xy["dz"]))
        ax_hist.axvline(med, color="orange", ls="--", label=f"median |Δz| = {med:.1f} µm")
        ax_hist.legend(fontsize=10)
    ax_hist.set(
        xlabel="|Δz| (µm)",
        ylabel="cells",
        title=f"Depth offset for cells within {XY_RADIUS:g} µm in XY (n={len(close_xy):,})",
    )
    plt.close(fig)
    mo.vstack(
        [
            mo.md(
                f"**{len(seeds):,}** seed cells from **{sel}**. "
                f"Among cells within {XY_RADIUS:g} µm in XY, "
                f"**{len(lying):,}** sit more than {Z_GAP:g} µm above or below their flat-map neighbor."
            ),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
