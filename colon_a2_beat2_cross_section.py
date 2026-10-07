"""Beat 2: cut like a pathologist — Cross-section through depth."""

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
    # Beat 2 — Cut like a pathologist

    **Flatten:** a single XY plane breaks epithelial walls into **arcs** and hides whether stroma
    and lumen are truly continuous through the section.

    **Collapse:** in the Inspect cube, turn on **Cross-section** and sweep through epithelium
    versus stroma or lumen. Walls that looked like broken rings on the map stay **continuous
    through depth** — the cut follows tissue, not a flat projection. **Save** the inspect window
    when you want that volume as a selection for later beats.

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
