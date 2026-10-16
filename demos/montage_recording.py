# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "milume==1.3.5",
#     "marimo>=0.25.1",
#     "spatialdata>=0.8.0",
#     "scanpy>=1.12",
#     "matplotlib",
#     "pandas",
#     "pyarrow",
# ]
#
# [tool.marimo.display]
# theme = "dark"
# code_editor_font_size = 20
#
# [tool.marimo.save]
# autosave = "off"
# ///
"""Ready-to-record notebook for the Milume montage (colon A2, milume 1.3.5).

Run from the repo root:

    uvx marimo edit --sandbox demos/montage_recording.py

If your marimo config doesn't auto-run cells, click Run (bottom right)
once after it opens. Data defaults to ``data/colon_a2.sdata.zarr`` (repo-relative); set
``COLON_A2_SDATA`` to read another store.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="columns",
    app_title="Milume montage",
    css_file="montage_recording.css",
)


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Shot list": mo.md(
                """
    *Before recording: if cells show as stale, click ▶ (bottom right)
    once so the data loads and the chart cell shows its placeholder.*

    1. **Opener.** Default points view; slow drift (scroll-zoom or pan).
    2. **Inspect.** Inspect tool, hover, click to place the cube, slide
       the cross-section one way on 2 axes, orbit, zoom.
       *Labels stay at the 0.40 default; Points is off by default,
       leave it off.*
    3. **Landmark.** Zoom to the top-left crypts, draw one shape
       around them (right-click the landmark tool → Shape), then Buffer in
       the context toolbar → turn off "Both sides" → drag the slider.
    4. **Probe.** *Tip: right panel → embed tab; check it shows `X_pca`.*
       Press **P** and hover.
    5. **Notebook.** Type `w = milume.peek(sdata)` in the empty cell,
       run it, then lasso; the chart on the right redraws.
    6. **End card** (Designer).
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    # colon_a2_common lives at the repo root, one level up.
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    os.environ.setdefault(
        "COLON_A2_SDATA", str(_root / "data" / "colon_a2.sdata.zarr")
    )

    import contextlib
    import io
    import logging
    import warnings

    import matplotlib.pyplot as plt
    import milume
    import scanpy as sc

    import colon_a2_common as common

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    common.apply_mpl_theme("dark")
    _quiet = io.StringIO()
    with contextlib.redirect_stdout(_quiet), contextlib.redirect_stderr(_quiet):
        # Annotations joined (cell_type, lineage, domain) and log-normalized.
        sdata, adata, _groups, _markers = common.load_colon_a2()
        # PCA embedding for the Probe tool (obsm["X_pca"]).
        sc.pp.pca(adata, n_comps=30)
    return adata, milume, mo, plt, sdata


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo, plt):
    tick = mo.ui.refresh(default_interval="1s")

    def bar_chart(cell_types):
        """Cell-type composition: top 7 plus "other"."""
        counts = cell_types.value_counts()
        top = counts.head(7)
        top["other"] = counts[7:].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(
            top.index[::-1].astype(str), top[::-1], color="#0d9488"
        )
        ax.bar_label(bars, fmt="{:,.0f}")
        ax.set_title(f"{len(cell_types):,} cells (sum {top.sum():,})")
        fig.tight_layout()
        return fig

    mo.Html(f'<div style="height:0;overflow:hidden">{tick}</div>')
    return bar_chart, tick


@app.cell
def _(adata, bar_chart, mo, tick):
    tick.value  # re-checks the lasso every second
    _w = globals().get("w")
    sel = _w.selections[-1:] if _w is not None else []
    if sel:
        names = _w.get_obs_names(selection_id=sel[0]["id"])
        out = bar_chart(adata[names].obs["cell_type"])
    else:
        out = mo.md("_Lasso a region in the widget._")
    out
    return


if __name__ == "__main__":
    app.run()
