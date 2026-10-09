"""Spatial analysis of colorectal cancer with 3D context (Glasgow colon A2).

Loads the SpatialData that ``build_colon_a2.py`` builds into ``data/`` (not committed)
from the ``colon/`` folder of the Stellaromics/demo dataset on Hugging Face. The Milume
widget comes first; each vignette below reads the landmarks you draw in it and runs a
short scverse-style analysis. Helpers live in ``colon_a2_common.py``.

    uv run python build_colon_a2.py --download --overwrite
    uv run marimo edit colon_a2.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import colon_a2_common as common
    from colon_a2_common import CELL_TYPE, apply_mpl_theme, load_colon_a2, tool_icon
    from milume import LandmarksWidget, landmarks_to_geodataframe

    return (
        CELL_TYPE,
        LandmarksWidget,
        apply_mpl_theme,
        common,
        landmarks_to_geodataframe,
        load_colon_a2,
        mo,
        tool_icon,
    )


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(common, mo):
    mo.md(f"""
    # Spatial analysis of colorectal cancer with 3D context

    A 1020-plex spatial transcriptomics section (~5 mm × 4 mm × 140 µm) of colorectal cancer
    on the Stellaromics Pyxa platform (`{common.SDATA_PATH.name}`, built by `build_colon_a2.py`).
    Draw landmarks in the **Milume** widget below; each vignette reads the ones you draw.

    1. **Shape → composition by depth**: what a flat composition bar hides.
    2. **Line + wide buffer → expression gradients** along and across a boundary.
    """)
    return


@app.cell
def _(load_colon_a2):
    sdata, adata, groups, marker_genes = load_colon_a2()
    return adata, groups, marker_genes, sdata


@app.cell(expand_output=True)
def _(CELL_TYPE, LandmarksWidget, marker_genes, mo, sdata):
    widget = LandmarksWidget(sdata, color=CELL_TYPE, genes=marker_genes, contrast_limits=(40, 255))
    landmarks = mo.ui.anywidget(widget)
    landmarks
    return (landmarks,)


@app.cell(hide_code=True)
def _(mo, tool_icon):
    _tools = [
        (["mouse-pointer-2"], "Select (V)", "Click a landmark or selection to edit it; drag its vertices."),
        (["hand"], "Move (H)", "Pan and zoom the map."),
        (["box"], "Inspect (I)", "Click the map to open a 300 µm 3D cube of cells and image there; orbit it."),
        (["lasso"], "Lasso (L)", "Draw a free-form selection of cells."),
        (["dot", "pentagon"], "Shape landmark (4)", "From the landmark menu: outline a region. Vignette 1 reads it."),
        (["dot", "move-up-right"], "Line landmark (2)", "From the landmark menu: draw a line. Vignette 2 reads it."),
        (["circle-dot-dashed"], "Buffer", "In a selected line's toolbar: widen the band of cells around the line."),
    ]
    _rows = "".join(
        '<div style="display:contents">'
        f'<span style="display:inline-flex;gap:0.3em;white-space:nowrap">{"".join(tool_icon(i) for i in icons)}</span>'
        f"<b>{name}</b><span>{text}</span></div>"
        for icons, name, text in _tools
    )
    mo.vstack([
        mo.md("### Widget tools\nThe toolbar at the top of the map; hover a button for its name."),
        mo.Html(
            '<div style="display:grid;grid-template-columns:max-content max-content 1fr;'
            f'gap:0.45em 1.2em;align-items:center">{_rows}</div>'
        ),
    ])
    return


@app.cell
def _(mo):
    get_shape, set_shape = mo.state(None)
    get_line, set_line = mo.state(None)
    get_sel1, set_sel1 = mo.state("all")
    return get_line, get_sel1, get_shape, set_line, set_sel1, set_shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · Shape → composition by depth

    **Takeaway: a flat composition bar averages ~140 µm of tissue; 5 µm z bins show which
    cell types are stacked at which depth inside the same outline.**

    Draw a **shape** landmark around a group of cells, for example a field of normal crypts.
    Open **Inspect** inside it to orbit the same cells in 3D, and **Save** the cube to restrict
    the analysis to it under **Cells**.
    """)
    return


@app.cell
def _(common, get_sel1, get_shape, groups, landmarks, mo, set_sel1, set_shape):
    _shapes = common.user_landmarks(landmarks.landmarks, ("shape",))
    _sels = ["all"] + [str(s["id"]) for s in landmarks.selections]
    shape_pick = mo.ui.dropdown(
        options=_shapes or ["(none)"],
        value=get_shape() if get_shape() in _shapes else (_shapes or ["(none)"])[0],
        label="Shape",
        on_change=set_shape,
    )
    sel1_pick = mo.ui.dropdown(
        options=_sels, value=get_sel1() if get_sel1() in _sels else "all", label="Cells", on_change=set_sel1
    )
    group1_pick = mo.ui.dropdown(options=groups, value="cell type", label="Group by")
    return group1_pick, sel1_pick, shape_pick


@app.cell
def _(adata, common, group1_pick, landmarks, landmarks_to_geodataframe, mo, sel1_pick, shape_pick):
    _gdf = landmarks_to_geodataframe([lm for lm in landmarks.landmarks if str(lm["id"]) == shape_pick.value])
    if len(_gdf) == 0:
        _out = mo.callout(
            mo.md("**No shape yet.** Pick **Shape** from the landmark menu (●, or press 4) and outline "
                  "a group of cells on the map; the composition by depth appears here."),
            kind="info",
        )
    else:
        _cells = None if sel1_pick.value == "all" else landmarks.get_obs_names(adata, selection_id=sel1_pick.value)
        _fig, _ = common.composition_by_z(adata, _gdf, group1_pick.value, _cells)
        _out = _fig if _fig is not None else mo.callout(
            mo.md(f"**{shape_pick.value}** covers no cells{'' if _cells is None else ' in ' + sel1_pick.value}. "
                  "Move or redraw it over tissue, or set **Cells** back to `all`."),
            kind="warn",
        )
    mo.vstack([mo.hstack([shape_pick, sel1_pick, group1_pick], justify="start"), _out], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · Line + wide buffer → expression gradients

    **Takeaway: along the line, epithelial genes give way to stromal ones where it crosses a
    boundary; across the line, flat profiles mean the boundary runs square to it, and a crossing
    off zero means it runs at an angle.**

    Draw a **line** landmark across a boundary, for example from tumour into stroma, and give it
    a wide buffer (≈150 µm) with **Buffer** in its toolbar. Cells in the buffer are projected onto
    the line: *along* is the distance from its start, *across* the signed perpendicular distance.
    Leave **Genes** empty to plot the two most rising and two most falling genes along the line.

    /// note
    Coloring the Inspect cube by a gene, to see the same gradient in 3D, is
    [milume#94](https://github.com/ckmah/milume/issues/94).
    ///
    """)
    return


@app.cell
def _(adata, common, get_line, landmarks, mo, set_line):
    _lines = common.user_landmarks(landmarks.landmarks, ("line", "spline"), buffered=True)
    line_pick = mo.ui.dropdown(
        options=_lines or ["(none)"],
        value=get_line() if get_line() in _lines else (_lines or ["(none)"])[0],
        label="Line",
        on_change=set_line,
    )
    gene_pick = mo.ui.multiselect(options=list(adata.var_names), value=[], label="Genes", max_selections=6)
    return gene_pick, line_pick


@app.cell
def _(adata, common, gene_pick, landmarks, landmarks_to_geodataframe, line_pick, mo):
    _gdf = landmarks_to_geodataframe([lm for lm in landmarks.landmarks if str(lm["id"]) == line_pick.value])
    if len(_gdf) == 0:
        _unbuffered = common.user_landmarks(landmarks.landmarks, ("line", "spline"))
        _out = mo.callout(
            mo.md(
                f"**{_unbuffered[0]}** has no buffer yet. Select it and set a wide buffer with **Buffer** in its toolbar."
                if _unbuffered
                else "**No buffered line yet.** Pick **Line** from the landmark menu (●, or press 2), draw it "
                "across a boundary, then set a wide buffer with **Buffer** in its toolbar."
            ),
            kind="info",
        )
    else:
        _coords = common.line_coordinates(adata, _gdf)
        if len(_coords) < 20:
            _out = mo.callout(
                mo.md(f"Only {len(_coords)} cells in the buffer of **{line_pick.value}**. "
                      "Widen the buffer or move the line over tissue."),
                kind="warn",
            )
        else:
            _genes = list(gene_pick.value) or common.gradient_genes(adata, _coords)
            _out = mo.vstack([
                mo.md(f"**{len(_coords):,}** cells in the buffer of **{line_pick.value}**; genes: {', '.join(_genes)}."),
                common.gradient_plot(adata, _coords, _genes),
            ])
    mo.vstack([mo.hstack([line_pick, gene_pick], justify="start"), _out], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    /// admonition | Acknowledgements
    **School of Cancer Sciences, University of Glasgow, UK**: Marta Campillo Poveda, Anthony Chalmers, Yoana Doncheva, Joanne Edwards, Andrea Gonzalez Ciscar, **Nigel Jamieson**, Claire Kennedy Dietrich, Ghazal Latif, Assya Legrini, Josefina Marinez Vasquez, Pamela McCall, Mari-Claire McGuigan, Luke McNickle, Tengyu Zhang

    **University of Edinburgh, UK**: Gerry Thompson

    **Stellaromics Inc, Boston, MA, USA**: Leah Carlson, Jeremy Lambert, Clarence Mah, Raghav Padmanabhan, Chan Park, Daphne Sze, Alexis Wong
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
