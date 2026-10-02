"""Glasgow colon A2: scverse analysis of a Pyxa SpatialData, steered by landmarks.

Loads the SpatialData that ``build_colon_a2.py`` builds into ``data/`` (not
committed) from the ``colon/`` folder of the Stellaromics/demo dataset on Hugging Face:
the full-section cell table in Pyxa µm, the 3D cell labels and Meteor's 3D nuclear
mosaic on one voxel grid. Two per-cell annotations, committed in
``annotations/colon_a2/``, are joined onto the table by ``cell_id``: cell types and lineages
(``cell_typing.parquet``) and Novae spatial domains (``novae_domains.parquet``).
``LandmarksWidget(sdata)`` shows them on the map, with cell-type markers in its gene
picker. Landmarks drawn there feed the ``milume`` measures (XY geometry, 1 µm z
bins), whose columns go back into ``adata.obs`` for scanpy plots and differential
expression.

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
    # Case Study: colorectal cancer

    Here we show a 3D spatial transcriptomics dataset with a 1020-plex gene panel profiling a colorectal cancer thick tissue section acquired on the Stellaromics Pyxa platform. The captured tissue volume is roughly 5mm x 4mm x 0.14mm (xyz).

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
    ## 1. Cell type and spatial domain annotations

    - `cell types:` 23 types, including a low-signal QC class
    - `lineage:` 5 coarse groups of cell types
    - `domain:` 10 Novae inferred spatial domains (L10), named by their niche signatures

    All annotations also have an additional "unassigned" class for unlabeled cells.

    /// admonition | Out of scope
    Cell typing and spatial domain inference (Novae) results were performed separately and pulled in here to keep this notebook focused on the case study.
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
    return domain_order, groups


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


@app.cell
def _(adata, groups, plt, sc):
    fig_umaps, axes_umap = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for _ax, (_name, _key) in zip(axes_umap, groups.items()):
        sc.pl.embedding(
            adata, "X_umap", color=_key, title=_name, size=1, frameon=False,
            legend_fontsize=11, ax=_ax, show=False,
        )
    plt.close(fig_umaps)
    fig_umaps
    return


@app.cell
def _(CELL_TYPE, DOMAIN, UNASSIGNED, adata, domain_order, pd, plt):
    MIN_SHARE = 0.02  # cell types below this share in every domain are merged into "Other"
    OTHER = f"Other (<{MIN_SHARE:.0%})"

    domain_counts = pd.crosstab(adata.obs[DOMAIN], adata.obs[CELL_TYPE])
    domain_counts = domain_counts.drop(index=UNASSIGNED, columns=UNASSIGNED)
    domain_share = domain_counts.div(domain_counts.sum(axis=1), axis=0)

    palette = dict(zip(adata.obs[CELL_TYPE].cat.categories, adata.uns[f"{CELL_TYPE}_colors"]))
    palette[OTHER] = "#8c8c8c"
    common = domain_share.columns[(domain_share >= MIN_SHARE).any()]
    bar_data = domain_share[common].assign(**{OTHER: domain_share.drop(columns=common).sum(axis=1)})
    # bars follow the similarity order that also sets the domain colors
    bar_data = bar_data.loc[domain_order]
    bar_data.index = [f"{d} (n={domain_counts.loc[d].sum():,})" for d in bar_data.index]

    fig_bars, ax_bars = plt.subplots(figsize=(15, 7), layout="constrained")
    bar_data.plot.barh(ax=ax_bars, stacked=True, width=0.85, color=[palette[c] for c in bar_data.columns])
    ax_bars.invert_yaxis()
    ax_bars.set(xlim=(0, 1), xlabel="share of the domain's cells", title="Cell types in each Novae domain")
    ax_bars.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    plt.close(fig_bars)
    fig_bars
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    ## 2. Blurring the lines between analysis and visualization

    Ever wished you could point at a structure in your tissue and just *measure* it? The colorectal cancer thick tissue section is 100um x 5 mm × 4 mm, so a 2D plot hides a lot.

    Introducing  <img src="https://raw.githubusercontent.com/ckmah/milume/6cf9f0660bc78bc7f18bf581f35c98e412af3ce3/assets/logo/favicon.svg" width="20" height="20" style="display:inline; padding: 0; margin:0;" alt="Link"> `milume`: a thinking surface for spatial omics. Milume is a reactive `anywidget` for engaging with spatial omics data, which then drives the analysis that follows, creating an efficient feedback loop for ideation.

    ### Analysis Flow
    """),
            mo.mermaid("""
    %%{init: {"theme": "base", "themeVariables": {"lineColor": "#888888", "textColor": "#888888", "edgeLabelBackground": "transparent", "clusterBkg": "transparent", "clusterBorder": "#888888"}, "flowchart": {"nodeSpacing": 30, "rankSpacing": 50, "curve": "basis"}}}%%
    flowchart LR
        A("<b>Data</b><br/>AnnData + SpatialData") --> B("<b>Widget</b><br/>LandmarksWidget")
        B --> C("<b>Outputs</b><br/>landmarks, selections, obs masks")
        C --> D("<b>Analysis</b><br/>measures and scanpy")
        D --> B
        classDef data fill:#6b7280,stroke:#9ca3af,color:#ffffff
        classDef ui fill:#2563eb,stroke:#60a5fa,color:#ffffff
        classDef out fill:#d97706,stroke:#fbbf24,color:#ffffff
        classDef py fill:#16a34a,stroke:#4ade80,color:#ffffff
        class A data
        class B ui
        class C out
        class D py
    """),
        ]
    )
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Measure cells and gene expression shifts in 2D and 3D

    The landmarks and selections made in the widget can now be accessed by code, allowing the user's knowledge about the tissue sample to drive downstream analysis.
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
    ## 4. A selection against the rest

    What makes a region different? Choose a selection in **Cells** above (a lasso, a promoted neighborhood
    or a saved inspect cube) and scanpy ranks the genes that set it apart from every other cell in the section.
    Membership is stored in `adata.obs["in_selection"]`, so you can reuse it anywhere.
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
