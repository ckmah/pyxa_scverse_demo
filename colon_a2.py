"""Glasgow colon A2: why 3D matters for a thick Pyxa SpatialData section.

Loads the SpatialData that ``build_colon_a2.py`` builds into ``data/`` (not
committed) from the ``colon/`` folder of the Stellaromics/demo dataset on Hugging Face.
Five notebook beats contrast what a flat 2D view implies with what 3D inspection
and measurement show: rings vs tubes, pathologist cuts, XY vs 3D neighbors,
composition by depth, and selection-driven DE.

    uv run python build_colon_a2.py --download --overwrite
    uv run marimo edit colon_a2.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import spatialdata as sd

    from milume import (
        LandmarksWidget,
        along_positions,
        composition,
        distances,
        landmarks_to_geodataframe,
        nearest_distances,
        write_obs,
    )
    from milume.categories import default_categorical_palette

    ROOT = Path(__file__).resolve().parent
    SDATA_PATH = ROOT / "data" / "colon_a2.sdata.zarr"
    CELL_TYPING = ROOT / "annotations" / "colon_a2" / "cell_typing.parquet"
    NOVAE_DOMAINS = ROOT / "annotations" / "colon_a2" / "novae_domains.parquet"
    NICHE_SIGNATURES = ROOT / "annotations" / "colon_a2" / "niche_signatures.csv"  # domain id -> niche name
    CELL_TYPE, LINEAGE, DOMAIN = "cell_type", "lineage", "domain"
    DOMAIN_LEVEL = "domain_L10"  # Novae resolution: L3, L5, L7, L10, L14 or L18
    UNASSIGNED = "unassigned"
    Z_BIN = 1.0  # µm; the measures bin depth at this width
    return (
        CELL_TYPE,
        CELL_TYPING,
        DOMAIN,
        DOMAIN_LEVEL,
        LINEAGE,
        LandmarksWidget,
        NICHE_SIGNATURES,
        NOVAE_DOMAINS,
        SDATA_PATH,
        UNASSIGNED,
        Z_BIN,
        along_positions,
        composition,
        default_categorical_palette,
        distances,
        landmarks_to_geodataframe,
        matplotlib,
        mo,
        nearest_distances,
        np,
        pd,
        plt,
        sc,
        sd,
        write_obs,
    )


@app.cell
def _(matplotlib, mo):
    theme = mo.app_meta().theme
    matplotlib.style.use("dark_background" if theme == "dark" else "default")
    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Why 3D matters: Glasgow colon A2

    A 1020-plex spatial transcriptomics section (~5 mm × 4 mm × 140 µm) from colorectal
    cancer tissue on the Stellaromics Pyxa platform. For tool builders, cell biologists,
    pathologists, biomedical researchers, and platform folks: each beat below contrasts what
    a **flat 2D map implies** with what **3D inspection and measurement** show.

    1. **Rings aren't rings** — crypt-like rings on the map are tubes through depth.
    2. **Cut like a pathologist** — cross-sections reveal walls continuous through Z.
    3. **Neighbors lie in 2D** — XY neighbors can be microns apart in Z.
    4. **Stacked niches** — composition changes with depth, not one flat mix.
    5. **See → analyze** — 3D picks the right cells for downstream DE.

    /// admonition | Acknowledgements
    **School of Cancer Sciences, University of Glasgow, UK**: Marta Campillo Poveda, Anthony Chalmers, Yoana Doncheva, Joanne Edwards, Andrea Gonzalez Ciscar, **Nigel Jamieson**, Claire Kennedy Dietrich, Ghazal Latif, Assya Legrini, Josefina Marinez Vasquez, Pamela McCall, Mari-Claire McGuigan, Luke McNickle, Tengyu Zhang

    **University of Edinburgh, UK**: Gerry Thompson

    **Stellaromics Inc, Boston, MA, USA**: Leah Carlson, Jeremy Lambert, Clarence Mah, Raghav Padmanabhan, Chan Park, Daphne Sze, Alexis Wong
    ///
    """)
    return


@app.cell
def _(SDATA_PATH, sd):
    sdata = sd.read_zarr(SDATA_PATH)
    adata = sdata["rna"]
    sdata
    return adata, sdata


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Trust the labels (short warmup)

    Per-cell annotations join by `cell_id`: 23 cell types, 5 lineages, and 10 Novae
    spatial domains (L10, named by niche signature). Two markers per typed cell are
    enough to sanity-check the labels before the 3D beats.

    /// admonition | Out of scope
    Cell typing and Novae domain inference were run separately; this notebook focuses on
    what 3D adds to interpretation.
    ///
    """)
    return


@app.cell
def _(
    CELL_TYPE,
    CELL_TYPING,
    DOMAIN,
    DOMAIN_LEVEL,
    LINEAGE,
    NICHE_SIGNATURES,
    NOVAE_DOMAINS,
    UNASSIGNED,
    adata,
    default_categorical_palette,
    matplotlib,
    mo,
    np,
    pd,
    plt,
):
    import colorsys

    from scipy.cluster.hierarchy import leaves_list, linkage

    typing = pd.read_parquet(CELL_TYPING, columns=["cell_id", CELL_TYPE, LINEAGE])
    domains = pd.read_parquet(NOVAE_DOMAINS, columns=["cell_id", DOMAIN_LEVEL])
    joined = (
        typing.merge(domains, on="cell_id", how="outer")
        .rename(columns={DOMAIN_LEVEL: DOMAIN})
        .set_index("cell_id")
        .reindex(adata.obs["cell_id"].astype(str))
    )
    joined[DOMAIN] = joined[DOMAIN].replace("nan", None)  # Novae left these unassigned
    niche_names = pd.read_csv(NICHE_SIGNATURES).set_index("domain")["name"]
    joined[DOMAIN] = joined[DOMAIN].map(niche_names)  # rename domain ids to niche names

    def by_size(values, last=()):
        order = list(values.value_counts().index)
        return [v for v in order if v not in last] + [v for v in last if v in order]

    for key, last in [(CELL_TYPE, ["Low-signal (QC)"]), (LINEAGE, []), (DOMAIN, [])]:
        categories = [*by_size(joined[key].dropna(), last), UNASSIGNED]
        adata.obs[key] = pd.Categorical(
            joined[key].fillna(UNASSIGNED).to_numpy(), categories=categories
        )
        adata.uns[f"{key}_colors"] = [
            *default_categorical_palette(len(categories) - 1),
            "#8c8c8c",
        ]
    # Order domains by similarity of their cell-type composition, and color them along that order,
    # so similar domains get similar colors.
    _share = pd.crosstab(joined[DOMAIN], joined[CELL_TYPE], normalize="index")
    domain_order = list(_share.index[leaves_list(linkage(_share, "average", metric="braycurtis", optimal_ordering=True))])
    def _cap_lightness(rgba, cap=0.6):
        """Darken pale colors (the middle of Spectral) so every domain stays visible."""
        h, l, s = colorsys.rgb_to_hls(*rgba[:3])
        return colorsys.hls_to_rgb(h, min(l, cap), s)

    _ramp = dict(zip(domain_order, map(_cap_lightness, plt.cm.Spectral_r(np.linspace(0.05, 0.95, len(domain_order))))))
    adata.uns[f"{DOMAIN}_colors"] = [
        *(matplotlib.colors.to_hex(_ramp[d]) for d in adata.obs[DOMAIN].cat.categories[:-1]),
        "#8c8c8c",
    ]
    groups = {"cell type": CELL_TYPE, "lineage": LINEAGE, "Novae domain": DOMAIN}

    n_typed = int((adata.obs[CELL_TYPE] != UNASSIGNED).sum())
    n_domain = int((adata.obs[DOMAIN] != UNASSIGNED).sum())
    mo.md(
        f"**{n_typed:,}** of {adata.n_obs:,} cells have a cell type and "
        f"**{n_domain:,}** a Novae domain ({DOMAIN_LEVEL}: "
        f"{len(adata.obs[DOMAIN].cat.categories) - 1} domains)."
    )
    return (groups,)


@app.cell
def _(UNASSIGNED, adata, groups, sc):
    MIN_EXPRESSING = 0.25  # a marker is detected in at least this share of the type's cells
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    cell_type = groups["cell type"]
    typed = [
        c for c in adata.obs[cell_type].cat.categories if c not in (UNASSIGNED, "Low-signal (QC)")
    ]
    sc.tl.rank_genes_groups(adata, cell_type, groups=typed, method="t-test", pts=True)
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    expressed = ranked[ranked["pct_nz_group"] >= MIN_EXPRESSING]
    top = expressed.sort_values("logfoldchanges", ascending=False).groupby("group", observed=True)
    by_type = top.head(2).groupby("group", observed=True)["names"].apply(list)
    marker_genes = list(dict.fromkeys(g for t in typed if t in by_type for g in by_type[t]))
    return (marker_genes,)


@app.cell
def _(CELL_TYPE, UNASSIGNED, adata, marker_genes, plt, sc):
    dotplot = sc.pl.dotplot(
        adata[adata.obs[CELL_TYPE] != UNASSIGNED],
        marker_genes,
        groupby=CELL_TYPE,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    dotplot.make_figure()
    plt.close("all")
    dotplot.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Beat 1 — Rings aren't rings

    On the flat cell-type map below, epithelial crypts read as **rings** or nested arcs.
    That is what a 2D projection *implies*.

    **Try it:** press **I** (Inspect), click a crypt-like field, and **orbit the cube**.
    The same structure is a **tube through Z**, not a flat ring — walls and lumen continue
    above and below the plane you were looking at.

    <img src="https://raw.githubusercontent.com/ckmah/milume/6cf9f0660bc78bc7f18bf581f35c98e412af3ce3/assets/logo/favicon.svg" width="20" height="20" style="display:inline; padding: 0; margin:0;" alt="milume"> `LandmarksWidget` colors cells by type (switch to lineage or domain in the picker) and packs marker genes for the gene picker.
    """)
    return


@app.cell(expand_output=True)
def _(CELL_TYPE, LandmarksWidget, marker_genes, mo, sdata):
    widget = LandmarksWidget(
        sdata, color=CELL_TYPE, genes=marker_genes, contrast_limits=(40, 255)
    )
    landmarks = mo.ui.anywidget(widget)
    landmarks
    return (landmarks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Beat 2 — Cut like a pathologist

    A single XY plane would break epithelial walls into **arcs** and hide whether stroma
    and lumen are truly continuous through the section.

    **Try it:** in the Inspect cube, turn on **Cross-section** and sweep through
    epithelium versus stroma or lumen. Walls that looked like broken rings on the map
    stay **continuous through depth** — the cut follows tissue, not a flat projection.
    **Save** the inspect window when you want that volume as a selection for later beats.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Beat 3 — Neighbors lie in 2D

    Lasso and neighborhood tools work on the **flat map**. A cell that looks like a
    neighbor in XY may sit **microns above or below** its nearest seed in Z.

    Pick a selection in **Cells** (a lasso, promoted neighborhood, or saved inspect cube).
    `milume.nearest_distances` pairs every cell with its nearest seed in XY and reports
    the depth offset `dz` to that seed — the gap a 2D view cannot see.
    """)
    return


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
    XY_RADIUS = 20.0  # µm; "looks like a neighbor" on the flat map
    Z_GAP = 10.0  # µm; meaningful depth separation in this section

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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Beat 4 — Stacked niches

    A single composition bar **collapses depth** into one mix. Draw a shape (or a buffered
    line) over a region and choose **Composition by depth**: cell-type, lineage, or domain
    proportions in 1 µm z bins show **stacked niches** — what a flat summary hides.
    """)
    return


@app.cell
def _(
    Z_BIN,
    adata,
    along_positions,
    composition,
    distances,
    np,
    pd,
    plt,
    sc,
    write_obs,
):
    MIN_CELLS = 20  # bins with fewer cells are left blank

    def heatmap(ax, table, *, x_width=None, y_width=None, label=""):
        """Draw `table` (rows on y, columns on x); a width makes that axis numeric bin edges."""

        def edges(values, width):
            if width is None:
                return np.arange(len(values) + 1)
            v = np.asarray(values, dtype=float)
            return np.append(v, v[-1] + width)

        mesh = ax.pcolormesh(
            edges(table.columns, x_width),
            edges(table.index, y_width),
            table.to_numpy(dtype=float),
            cmap="magma",
        )
        if x_width is None:
            ax.set_xticks(np.arange(len(table.columns)) + 0.5, table.columns, rotation=90)
        if y_width is None:
            ax.set_yticks(np.arange(len(table.index)) + 0.5, table.index)
        ax.figure.colorbar(mesh, ax=ax, label=label, shrink=0.8)

    def z_edges(d):
        return np.arange(d["z_bin"].min(), d["z_bin"].max() + 1.5 * Z_BIN, Z_BIN)

    def composition_by_depth(gdf, cells, lid, key):
        comp = composition(adata, gdf, obs_key=key, obs_names=cells, z_bin_size=Z_BIN)
        if comp.empty:
            return None
        by_z = comp.pivot_table(index="z_bin", columns="group", values="proportion", fill_value=0)
        n_per_z = comp.groupby("z_bin")["n_total"].first()
        by_z = by_z.where(n_per_z.reindex(by_z.index) >= MIN_CELLS)
        by_z = by_z.reindex(
            index=z_edges(comp)[:-1],
            columns=[g for g in adata.obs[key].cat.categories if g in by_z.columns],
        )
        pooled = comp.groupby("group")["count"].sum().reindex(by_z.columns)
        fig, (ax_bar, ax_z) = plt.subplots(
            1, 2, figsize=(11, 5), width_ratios=[1, 2.2], layout="constrained"
        )
        ax_bar.barh(pooled.index[::-1], (pooled / pooled.sum())[::-1])
        ax_bar.set(xlabel="proportion, whole depth", title=f"{int(pooled.sum()):,} cells")
        heatmap(ax_z, by_z, y_width=Z_BIN, label="proportion in z bin")
        ax_z.set(ylabel="z (µm)", title=f"Composition per {Z_BIN:g} µm z bin · {lid}")
        return fig

    def groups_profile(d, value, edges, xlabel, title, key):
        rows = {}
        for group in adata.obs[key].cat.categories:
            counts, _ = np.histogram(d.loc[d["group"] == group, value], bins=edges)
            if counts.max() > 0:
                rows[group] = counts / counts.max()
        fig, (ax_g, ax_z) = plt.subplots(
            1, 2, figsize=(12, 5), width_ratios=[1.4, 1], layout="constrained"
        )
        heatmap(
            ax_g,
            pd.DataFrame(rows, index=edges[:-1]).T,
            x_width=edges[1] - edges[0],
            label="density, peak 1",
        )
        ax_g.set(xlabel=xlabel, title=title)
        zs = z_edges(d)
        counts, _, _ = np.histogram2d(d["z"], d[value], bins=[zs, edges])
        ax_z.pcolormesh(edges, zs, counts, cmap="magma")
        ax_z.set(xlabel=xlabel, ylabel="z (µm)", title=f"cells per {Z_BIN:g} µm z bin")
        return fig

    def genes_profile(d, value, edges, fmt, genes, obs_col, title, tick_every=1):
        """Bin `d[value]` into `adata.obs[obs_col]` and let scanpy average the genes per bin."""
        values = np.full(adata.n_obs, np.nan)
        values[d["point_index"].to_numpy()] = d[value].to_numpy()
        labels = [fmt.format(e) for e in edges[:-1]]
        bins = pd.cut(values, edges, labels=labels, right=False)
        counts = pd.Series(bins).value_counts()
        adata.obs[obs_col] = bins.remove_categories(counts.index[counts < MIN_CELLS])
        view = adata[adata.obs[obs_col].notna()]
        mp = sc.pl.matrixplot(
            view,
            genes,
            groupby=obs_col,
            standard_scale="var",
            swap_axes=True,
            title=title,
            colorbar_title="scaled mean\nexpression",
            figsize=(min(16, 3 + 0.12 * len(labels)), 1.5 + 0.3 * len(genes)),
            show=False,
            return_fig=True,
        )
        mp.make_figure()
        if tick_every > 1:
            for tick in mp.get_axes()["mainplot_ax"].get_xticklabels():
                tick.set_visible(float(tick.get_text()) % tick_every == 0)
        return mp.fig

    def in_region(d, gdf):
        """Cells inside a shape (distance 0) or a buffered band; None for other landmarks."""
        ltype = gdf["type"].iloc[0]
        if ltype == "shape":
            return d[d["distance"] == 0]
        if ltype in ("line", "spline") and gdf["buffer_width"].iloc[0] > 0:
            return d
        return None

    def run_measure(measure, gdf, cells, genes, key):
        """Figure for `measure` on one landmark, grouped by `obs[key]`, or a note on why there is none."""
        lid = str(gdf["id"].iloc[0])
        needs = f"**{measure}** needs {MEASURES[measure]} that covers the cells."
        if measure == "Composition by depth":
            return composition_by_depth(gdf, cells, lid, key) or needs
        along = "along path" in measure
        measure_fn = along_positions if along else distances
        d = measure_fn(adata, gdf, obs_key=key, obs_names=cells, z_bin_size=Z_BIN)
        if d.empty:
            return needs
        value = "s" if along else "distance"
        obs_col = "path_s" if along else f"dist_{lid}"
        write_obs(adata, d, obs_col, value)
        if along:
            edges, fmt, xlabel = np.linspace(0, 1, 21), "{:.2f}", "position along path, start → end"
        else:
            hi = max(float(d["distance"].quantile(0.98)), 1.0)
            edges, fmt, xlabel = np.linspace(0, hi, 25), "{:.0f}", "distance from landmark (µm)"
            d = d[d["distance"] < hi]
        title = f"{measure} · {lid}"
        if measure.startswith("Composition"):
            return groups_profile(d, value, edges, xlabel, title, key)
        if measure == "Genes by depth":
            region = in_region(d, gdf)
            if region is None or region.empty:
                return needs
            return genes_profile(
                region, "z", z_edges(region), "{:.0f}", genes, "z_bin", title, tick_every=10
            )
        return genes_profile(d, value, edges, fmt, genes, f"{obs_col}_bin", title)

    MEASURES = {
        "Composition by depth": "a shape, or a line / spline with a buffer,",
        "Composition vs distance": "a landmark",
        "Composition along path": "a line or spline",
        "Genes vs distance": "a landmark",
        "Genes along path": "a line or spline",
        "Genes by depth": "a shape, or a line / spline with a buffer,",
    }
    return MEASURES, run_measure


@app.cell
def _(MEASURES, adata, groups, marker_genes, mo):
    measure_pick = mo.ui.dropdown(
        options=list(MEASURES), value="Composition by depth", label="Measure"
    )
    group_pick = mo.ui.dropdown(options=groups, value="cell type", label="Group by")
    gene_pick = mo.ui.multiselect(
        options=list(adata.var_names),
        value=marker_genes[:8],
        label="Genes",
        full_width=True,
    )
    return gene_pick, group_pick, measure_pick


@app.cell
def _(
    adata,
    gene_pick,
    group_pick,
    landmark_pick,
    landmarks,
    landmarks_to_geodataframe,
    measure_pick,
    mo,
    plt,
    run_measure,
    selection_pick,
):
    gdf = landmarks_to_geodataframe(
        [lm for lm in landmarks.landmarks if str(lm["id"]) == landmark_pick.value]
    )
    if len(gdf) == 0:
        result = "Draw a landmark on the map to measure it."
    elif measure_pick.value.startswith("Genes") and not gene_pick.value:
        result = "Pick at least one gene."
    else:
        cells = landmarks.get_obs_names(adata, selection_id=selection_pick.value)
        result = run_measure(
            measure_pick.value, gdf, cells, list(gene_pick.value), group_pick.value
        )
    measured = mo.md(f"_{result}_") if isinstance(result, str) else result
    plt.close("all")
    return (measured,)


@app.cell(hide_code=True)
def _(
    gene_pick,
    group_pick,
    landmark_pick,
    measure_pick,
    measured,
    mo,
    selection_pick,
):
    picks = [landmark_pick, selection_pick, measure_pick]
    if measure_pick.value.startswith("Composition"):
        picks.append(group_pick)
    controls = [mo.hstack(picks, justify="start")]
    if measure_pick.value.startswith("Genes"):
        controls.append(gene_pick)
    mo.vstack([*controls, measured], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Beat 5 — See → analyze

    3D is not just prettier viewing — it is how you **pick the right cells** to analyze.
    Choose a selection in **Cells** (lasso, promoted neighborhood, or saved inspect cube).
    Scanpy ranks genes that set it apart from the rest of the section; membership lives in
    `adata.obs["in_selection"]` for any downstream step.
    """)
    return


@app.cell
def _(adata, group_pick, landmarks, mo, pd, sc, selection_pick):
    sel = selection_pick.value
    mo.stop(sel == "all", mo.md("_Pick a selection in **Cells** to compare it with the rest._"))
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
