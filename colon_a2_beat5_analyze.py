"""Beat 5: see → analyze — saved inspect selection to differential expression."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import scanpy as sc

    from colon_a2_common import CELL_TYPE, apply_mpl_theme, load_colon_a2
    from milume import LandmarksWidget

    return CELL_TYPE, LandmarksWidget, apply_mpl_theme, load_colon_a2, mo, pd, sc


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Beat 5 — See → analyze

    **Flatten:** 3D viewing alone does not change your analysis — you still need the right cells.

    **Collapse:** **Save** an inspect cube (or lasso a region), pick it in **Cells**, and scanpy
    ranks genes that set it apart from the rest of the section. Membership lives in
    `adata.obs["in_selection"]` — 3D is how you pick the right cells to analyze.

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
    get_selection, set_selection = mo.state("all")
    return get_selection, set_selection


@app.cell
def _(get_selection, groups, landmarks, mo, set_selection):
    selection_ids = ["all"] + [str(s["id"]) for s in landmarks.selections]
    selection_pick = mo.ui.dropdown(
        options=selection_ids,
        value=get_selection() if get_selection() in selection_ids else "all",
        label="Cells",
        on_change=set_selection,
    )
    group_pick = mo.ui.dropdown(options=groups, value="cell type", label="Summarize by")
    mo.hstack([selection_pick, group_pick], justify="start")
    return group_pick, selection_pick


@app.cell
def _(adata, group_pick, landmarks, mo, pd, sc, selection_pick):
    sel = selection_pick.value
    mo.stop(sel == "all", mo.md("_Pick a selection in **Cells** (saved inspect cube, lasso, or neighborhood)._"))
    landmarks.assign_obs_mask(adata, "in_selection", selection_id=sel)
    adata.obs["in_selection"] = pd.Categorical(
        adata.obs["in_selection"].map({True: "in", False: "out"}), categories=["in", "out"]
    )
    n_in = int((adata.obs["in_selection"] == "in").sum())
    mo.stop(n_in < 3, mo.md(f"_Selection **{sel}** holds {n_in} cells; draw a larger one._"))
    sc.tl.rank_genes_groups(
        adata, "in_selection", groups=["in"], reference="out", method="t-test", key_added="rank_selection"
    )
    de = sc.get.rank_genes_groups_df(adata, group="in", key="rank_selection").head(20)
    in_sel = adata.obs["in_selection"] == "in"
    share = adata.obs.loc[in_sel, group_pick.value].value_counts(normalize=True)
    mo.vstack(
        [
            mo.md(f"**{n_in:,}** cells in **{sel}**, mostly {', '.join(share.index[:3])}"),
            mo.ui.table(
                de[["names", "logfoldchanges", "pvals_adj", "scores"]].round(3),
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
