"""Glasgow colon A2: one widget, three short 3D vignettes.

Loads the SpatialData that ``build_colon_a2.py`` builds into ``data/`` (not committed)
from the ``colon/`` folder of the Stellaromics/demo dataset on Hugging Face. The Milume
widget comes first; each vignette below reads a landmark or selection from it and runs a
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
    import numpy as np

    import colon_a2_common as common
    from colon_a2_common import CELL_TYPE, LINEAGE, apply_mpl_theme, load_colon_a2
    from milume import LandmarksWidget, enrichment, landmarks_to_geodataframe

    return (
        CELL_TYPE,
        LINEAGE,
        LandmarksWidget,
        apply_mpl_theme,
        common,
        enrichment,
        landmarks_to_geodataframe,
        load_colon_a2,
        mo,
        np,
    )


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(common, mo):
    mo.md(f"""
    # Glasgow colon A2 in 3D: three vignettes

    A 1020-plex spatial transcriptomics section (~5 mm × 4 mm × 140 µm) of colorectal cancer
    on the Stellaromics Pyxa platform (`{common.SDATA_PATH.name}`, built by `build_colon_a2.py`).
    The **Milume** widget below is the starting point for every analysis: draw a landmark or
    make a selection there, then pick it in the vignette that uses it.

    1. **Shape → Inspect → composition by depth**: what a flat composition bar hides.
    2. **Immune cells → neighborhood → composition**: who sits next to immune cells.
    3. **Line + wide buffer → expression gradients** along and across a boundary.

    Each vignette starts from a demo landmark (`{common.DEMO_SHAPE}`, `{common.DEMO_LINE}`)
    already on the map, so the notebook runs end to end; draw your own and it takes over.
    """)
    return


@app.cell
def _(load_colon_a2):
    sdata, adata, groups, marker_genes = load_colon_a2()
    return adata, groups, marker_genes, sdata


@app.cell(expand_output=True)
def _(CELL_TYPE, LandmarksWidget, adata, common, marker_genes, mo, sdata):
    widget = LandmarksWidget(sdata, color=CELL_TYPE, genes=marker_genes, contrast_limits=(40, 255))
    widget.landmarks = common.default_landmarks(adata)
    _shape = widget.landmarks[0]["vertices"]
    widget.inspect_cx = (_shape[0][0] + _shape[2][0]) / 2
    widget.inspect_cy = (_shape[0][1] + _shape[2][1]) / 2
    landmarks = mo.ui.anywidget(widget)
    landmarks
    return (landmarks,)


@app.cell
def _(mo):
    get_shape, set_shape = mo.state(None)
    get_line, set_line = mo.state(None)
    get_sel1, set_sel1 = mo.state("all")
    get_sel2, set_sel2 = mo.state(None)
    return get_line, get_sel1, get_sel2, get_shape, set_line, set_sel1, set_sel2, set_shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · Shape → Inspect → composition by depth

    **Takeaway: a flat composition bar averages ~140 µm of tissue; 5 µm z bins show which
    cell types are stacked at which depth inside the same outline.**

    Draw a **shape** around a group of cells. The demo square sits on the cleanest dense
    normal-crypt field: the 300 µm square with the highest share of crypt stem/TA, goblet,
    colonocyte and enteroendocrine cells among those under 5% tumour epithelium; the Inspect cube
    starts centred on it. Press **I** and click inside a shape to open the Inspect cube and orbit
    the cells you outlined; **Save** the cube to restrict the analysis to it under **Cells**.
    """)
    return


@app.cell
def _(common, get_sel1, get_shape, groups, landmarks, mo, set_sel1, set_shape):
    _shapes = common.pick_landmark(landmarks.landmarks, ("shape",), common.DEMO_SHAPE)
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
        _out = mo.md("_No shape landmark on the map. Draw one (Shape tool) around a group of cells._")
    else:
        _cells = None if sel1_pick.value == "all" else landmarks.get_obs_names(adata, selection_id=sel1_pick.value)
        _fig, _ = common.composition_by_z(adata, _gdf, group1_pick.value, _cells)
        _out = _fig if _fig is not None else mo.md(
            f"_**{shape_pick.value}** covers no cells{'' if _cells is None else ' in ' + sel1_pick.value}. "
            "Move or redraw it over tissue, or set **Cells** back to `all`._"
        )
    mo.vstack([mo.hstack([shape_pick, sel1_pick, group1_pick], justify="start"), _out], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · Immune cells → neighborhood → composition

    **Takeaway: the cells within a few µm of immune cells are not a random sample of the
    tissue; bars above zero are cell types enriched next to the chosen immune cells.**

    In the widget, select immune cells, expand them with **Neighbors**, and promote the
    neighborhood to a selection; it shows up under **Neighborhood** (newest first) and is
    compared with all other non-seed cells. Pick the matching immune types under **Seed immune
    cell types** so the seeds themselves are left out of the neighborhood.

    /// note | Python fallback (remove once milume#92 lands)
    With no widget selection (or an empty one), the neighborhood is computed in Python: non-seed
    cells within **Radius** of a seed in 3D. It stands in for select cells → neighbors, which is
    slow on this section today ([milume#92](https://github.com/ckmah/milume/issues/92)); the
    multiselect stands in for picking several groups per category set
    ([milume#93](https://github.com/ckmah/milume/issues/93)).
    ///
    """)
    return


@app.cell
def _(CELL_TYPE, LINEAGE, adata, common, get_sel2, groups, landmarks, mo, set_sel2):
    FALLBACK = "(Python fallback)"
    _immune = sorted(
        set(adata.obs.loc[adata.obs[LINEAGE] == "Immune", CELL_TYPE].astype(str)) - {common.UNASSIGNED}
    )
    # Widget selections first (newest first); the Python fallback is last.
    _sels = [str(s["id"]) for s in reversed(landmarks.selections)] + [FALLBACK]
    seed_pick = mo.ui.multiselect(options=_immune, value=_immune, label="Seed immune cell types")
    radius_pick = mo.ui.slider(5, 50, step=5, value=15, label="Radius (µm)", show_value=True)
    sel2_pick = mo.ui.dropdown(
        options=_sels,
        value=get_sel2() if get_sel2() in _sels else _sels[0],
        label="Neighborhood",
        on_change=set_sel2,
    )
    group2_pick = mo.ui.dropdown(options=groups, value="cell type", label="Group by")
    return FALLBACK, group2_pick, radius_pick, sel2_pick, seed_pick


@app.cell
def _(CELL_TYPE, FALLBACK, adata, common, enrichment, group2_pick, landmarks, mo, np, radius_pick, sel2_pick, seed_pick):
    _controls = mo.hstack([seed_pick, radius_pick, sel2_pick, group2_pick], justify="start", wrap=True)
    _seeds = adata.obs[CELL_TYPE].isin(seed_pick.value).to_numpy()
    _names = np.asarray(adata.obs_names.astype(str))
    if not _seeds.any():
        _out = mo.md("_Pick at least one immune cell type as seeds._")
    else:
        # Primary path: the neighborhood the user promoted in the widget.
        _hood = np.zeros(adata.n_obs, dtype=bool)
        if sel2_pick.value != FALLBACK:
            # Hash join on obs_names: np.isin on object string arrays is quadratic at this size.
            _picked = landmarks.get_obs_names(adata, selection_id=sel2_pick.value)
            _hood = adata.obs_names.astype(str).isin(_picked) & ~_seeds
        _note = f" Neighborhood: widget selection **{sel2_pick.value}**."
        if not _hood.any():
            # Fallback until ckmah/milume#92 makes select cells -> neighbors fast enough on this
            # section: compute the neighborhood in Python. Remove this branch once #92 lands.
            _hood, _hood_xy = common.neighborhood(adata, _seeds, float(radius_pick.value))
            _flat_only = int((_hood_xy & ~_hood).sum())
            _note = (
                f" Neighborhood: Python fallback, {radius_pick.value} µm in 3D"
                f"{'' if sel2_pick.value == FALLBACK else f' (selection {sel2_pick.value} had no non-seed cells)'}."
                f" Another **{_flat_only:,}** cells are within {radius_pick.value} µm on the flat map"
                " but farther than that in 3D."
            )
        if not _hood.any():
            _out = mo.md("_The neighborhood is empty. Raise **Radius**, or promote a larger neighborhood in the widget._")
        else:
            _table = enrichment(adata, _names[_hood], obs_key=group2_pick.value, background=_names[~_seeds])
            _top = _table[np.isfinite(_table["log2_enrichment"])].head(3)["group"]
            _out = mo.vstack([
                mo.md(
                    f"**{int(_seeds.sum()):,}** seed cells, **{int(_hood.sum()):,}** neighbors; most enriched: "
                    f"{', '.join(_top)}.{_note}"
                ),
                common.enrichment_plot(_table, f"Neighborhood vs background · {group2_pick.value}"),
            ])
    mo.vstack([_controls, _out], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · Line + wide buffer → expression gradients

    **Takeaway: along the line, epithelial genes give way to stromal ones where it crosses a
    boundary; across the line, flat profiles mean the boundary runs square to it, and a crossing
    off zero means it runs at an angle.**

    Draw a **line** across a boundary and give it a wide buffer in the landmark toolbar (the
    demo line runs from tumour core into stroma with a 150 µm half-width). Cells in the buffer
    are projected onto the line: *along* is the distance from its start, *across* the signed
    perpendicular distance. Leave **Genes** empty to plot the two most rising and two most
    falling genes along the line.

    /// note
    Coloring the Inspect cube by a gene, to see the same gradient in 3D, is
    [milume#94](https://github.com/ckmah/milume/issues/94).
    ///
    """)
    return


@app.cell
def _(adata, common, get_line, landmarks, mo, set_line):
    _lines = common.pick_landmark(landmarks.landmarks, ("line", "spline"), common.DEMO_LINE)
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
        _out = mo.md("_No buffered line on the map. Draw a line, then set a buffer width in its toolbar._")
    else:
        _coords = common.line_coordinates(adata, _gdf)
        if len(_coords) < 20:
            _out = mo.md(
                f"_Only {len(_coords)} cells in the buffer of **{line_pick.value}**. "
                "Widen the buffer or move the line over tissue._"
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
