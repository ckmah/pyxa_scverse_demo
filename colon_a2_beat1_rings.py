"""Beat 1: rings aren't rings — flat map vs Inspect cube."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from colon_a2_common import CELL_TYPE, apply_mpl_theme, load_colon_a2
    from milume import LandmarksWidget

    return CELL_TYPE, LandmarksWidget, apply_mpl_theme, load_colon_a2, mo


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Beat 1 — Rings aren't rings

    **Flatten:** on the flat cell-type map below, epithelial crypts read as **rings** or nested arcs.

    **Collapse:** press **I** (Inspect), click a crypt-like field, and **orbit the cube** — the same
    structure is a **tube through Z**, not a flat ring. Walls and lumen continue above and below
    the plane you were looking at.

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
    return


if __name__ == "__main__":
    app.run()
