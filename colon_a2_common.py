"""Shared Glasgow colon A2 data load, annotation, and vignette helpers.

Used by ``colon_a2.py``. Assumes ``data/colon_a2.sdata.zarr`` exists (see
``build_colon_a2.py``); set ``COLON_A2_SDATA`` to read another store (the smoke test
points it at a small synthetic one).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from milume.categories import default_categorical_palette

ROOT = Path(__file__).resolve().parent
SDATA_PATH = Path(os.environ.get("COLON_A2_SDATA", ROOT / "data" / "colon_a2.sdata.zarr"))
CELL_TYPING = ROOT / "annotations" / "colon_a2" / "cell_typing.parquet"
NOVAE_DOMAINS = ROOT / "annotations" / "colon_a2" / "novae_domains.parquet"
NICHE_SIGNATURES = ROOT / "annotations" / "colon_a2" / "niche_signatures.csv"
CELL_TYPE = "cell_type"
LINEAGE = "lineage"
DOMAIN = "domain"
DOMAIN_LEVEL = "domain_L10"  # Novae resolution: L3, L5, L7, L10, L14 or L18
UNASSIGNED = "unassigned"
Z_BIN = 1.0  # µm; milume measures bin depth at this width
GROUPS = {"cell type": CELL_TYPE, "lineage": LINEAGE, "Novae domain": DOMAIN}


def apply_mpl_theme(theme: str | None = None) -> None:
    """Match matplotlib style to marimo light/dark theme."""
    if theme is None:
        theme = "default"
    matplotlib.style.use("dark_background" if theme == "dark" else "default")
    matplotlib.rcParams.update(
        {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11}
    )


def load_sdata(path: Path | None = None) -> sd.SpatialData:
    return sd.read_zarr(path or SDATA_PATH)


def join_annotations(adata) -> dict[str, str]:
    """Join cell typing and Novae domains onto ``adata.obs``; return group labels."""
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
    joined[DOMAIN] = joined[DOMAIN].replace("nan", None)
    niche_names = pd.read_csv(NICHE_SIGNATURES).set_index("domain")["name"]
    joined[DOMAIN] = joined[DOMAIN].map(niche_names)

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

    share = pd.crosstab(joined[DOMAIN], joined[CELL_TYPE], normalize="index")
    domain_order = list(
        share.index[
            leaves_list(linkage(share, "average", metric="braycurtis", optimal_ordering=True))
        ]
    )

    def cap_lightness(rgba, cap=0.6):
        h, l, s = colorsys.rgb_to_hls(*rgba[:3])
        return colorsys.hls_to_rgb(h, min(l, cap), s)

    ramp = dict(
        zip(
            domain_order,
            map(cap_lightness, plt.cm.Spectral_r(np.linspace(0.05, 0.95, len(domain_order)))),
        )
    )
    adata.uns[f"{DOMAIN}_colors"] = [
        *(matplotlib.colors.to_hex(ramp[d]) for d in adata.obs[DOMAIN].cat.categories[:-1]),
        "#8c8c8c",
    ]
    return dict(GROUPS)


def compute_marker_genes(adata, groups: dict[str, str] | None = None, min_expressing: float = 0.25):
    """Pick two markers per cell type for the widget gene picker."""
    groups = groups or GROUPS
    cell_type = groups["cell type"]
    typed = [
        c for c in adata.obs[cell_type].cat.categories if c not in (UNASSIGNED, "Low-signal (QC)")
    ]
    sc.tl.rank_genes_groups(adata, cell_type, groups=typed, method="t-test", pts=True)
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    expressed = ranked[ranked["pct_nz_group"] >= min_expressing]
    top = expressed.sort_values("logfoldchanges", ascending=False).groupby("group", observed=True)
    by_type = top.head(2).groupby("group", observed=True)["names"].apply(list)
    return list(dict.fromkeys(g for t in typed if t in by_type for g in by_type[t]))


def load_colon_a2(*, normalize: bool = True):
    """Load SpatialData, join annotations, normalize counts, and compute marker genes."""
    sdata = load_sdata()
    adata = sdata["rna"]
    groups = join_annotations(adata)
    marker_genes: list[str] = []
    if normalize:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        marker_genes = compute_marker_genes(adata, groups)
    return sdata, adata, groups, marker_genes


# --- Widget tools --------------------------------------------------------------------------
# Lucide glyphs (ISC license) that the Milume toolbar uses, so the intro matches the widget.
TOOL_ICONS = {
    "mouse-pointer-2": '<path d="M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z"/>',
    "hand": '<path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
    "box": '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "lasso": '<path d="M3.704 14.467a10 8 0 1 1 3.115 2.375"/><path d="M7 22a5 5 0 0 1-2-3.994"/><circle cx="5" cy="16" r="2"/>',
    "move-up-right": '<path d="M13 5H19V11"/><path d="M19 5L5 19"/>',
    "pentagon": '<path d="M10.83 2.38a2 2 0 0 1 2.34 0l8 5.74a2 2 0 0 1 .73 2.25l-3.04 9.26a2 2 0 0 1-1.9 1.37H7.04a2 2 0 0 1-1.9-1.37L2.1 10.37a2 2 0 0 1 .73-2.25z"/>',
    "circle-dot-dashed": '<path d="M10.1 2.18a9.93 9.93 0 0 1 3.8 0"/><path d="M17.6 3.71a9.95 9.95 0 0 1 2.69 2.7"/><path d="M21.82 10.1a9.93 9.93 0 0 1 0 3.8"/><path d="M20.29 17.6a9.95 9.95 0 0 1-2.7 2.69"/><path d="M13.9 21.82a9.94 9.94 0 0 1-3.8 0"/><path d="M6.4 20.29a9.95 9.95 0 0 1-2.69-2.7"/><path d="M2.18 13.9a9.93 9.93 0 0 1 0-3.8"/><path d="M3.71 6.4a9.95 9.95 0 0 1 2.7-2.69"/><circle cx="12" cy="12" r="1"/>',
    "dot": '<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>',
}


def tool_icon(name: str, size: int = 18) -> str:
    """Inline SVG for one widget toolbar glyph."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-0.25em">{TOOL_ICONS[name]}</svg>'
    )


def user_landmarks(landmarks: list[dict], kinds: tuple[str, ...], *, buffered: bool = False) -> list[str]:
    """Ids of visible landmarks of ``kinds`` drawn in the widget, newest first.

    ``buffered`` keeps only lines / splines with a buffer width > 0.
    """
    ids = [
        str(lm["id"])
        for lm in landmarks
        if lm.get("type") in kinds
        and not lm.get("hidden")
        and (not buffered or float(lm.get("buffer_width") or 0) > 0)
    ]
    return ids[::-1]


# --- Vignette 1: composition vs z --------------------------------------------------------
Z_PLOT_BIN = 5.0  # µm; composition-by-depth heatmap bins (1 µm bins are too sparse to read)


def composition_by_z(adata, gdf, key: str, obs_names=None, *, z_bin: float = Z_PLOT_BIN, min_cells: int = 10):
    """Flat composition bar next to a per-``z_bin`` heatmap for one shape landmark.

    Bins with fewer than ``min_cells`` cells are left blank.

    Returns ``(figure, by_z table)`` or ``(None, None)`` when the landmark covers no cells.
    """
    from milume import composition

    comp = composition(adata, gdf, obs_key=key, obs_names=obs_names, z_bin_size=z_bin)
    if comp.empty:
        return None, None
    by_z = comp.pivot_table(index="z_bin", columns="group", values="proportion", fill_value=0)
    n_per_z = comp.groupby("z_bin")["n_total"].first()
    by_z = by_z.where(n_per_z.reindex(by_z.index) >= min_cells)
    z_edges = np.arange(comp["z_bin"].min(), comp["z_bin"].max() + 1.5 * z_bin, z_bin)
    by_z = by_z.reindex(
        index=z_edges[:-1], columns=[g for g in adata.obs[key].cat.categories if g in by_z.columns]
    )
    pooled = comp.groupby("group")["count"].sum().reindex(by_z.columns)
    height = max(5.0, 0.3 * len(by_z.columns) + 1.5)
    fig, (ax_bar, ax_z) = plt.subplots(1, 2, figsize=(13, height), width_ratios=[1, 2.2], layout="constrained")
    ax_bar.barh(pooled.index[::-1], (pooled / pooled.sum())[::-1])
    ax_bar.set(xlabel="proportion, whole depth (flat summary)", title=f"{int(pooled.sum()):,} cells")
    edges = np.append(by_z.index.to_numpy(float), by_z.index[-1] + z_bin)
    mesh = ax_z.pcolormesh(np.arange(len(by_z.columns) + 1), edges, by_z.to_numpy(float), cmap="magma")
    ax_z.set_xticks(np.arange(len(by_z.columns)) + 0.5, by_z.columns, rotation=90)
    fig.colorbar(mesh, ax=ax_z, label="proportion in z bin", shrink=0.8)
    ax_z.set(ylabel="z (µm)", title=f"Composition per {z_bin:g} µm z bin · {gdf['id'].iloc[0]}")
    plt.close(fig)
    return fig, by_z


# --- Vignette 3: gradients along and across a line ----------------------------------------
def line_coordinates(adata, gdf) -> pd.DataFrame:
    """Cells in a buffered line landmark with ``along`` (µm from start) and signed ``across`` (µm).

    ``across`` > 0 is left of the line's direction of travel.
    """
    import shapely
    from milume import along_positions

    rows = along_positions(adata, gdf, obs_key=CELL_TYPE, z_bin_size=None)
    if rows.empty:
        return rows
    line = gdf.geometry.iloc[0]
    xy = np.asarray(adata.obsm["spatial"], dtype=float)[rows["point_index"].to_numpy(), :2]
    along = rows["s"].to_numpy() * line.length
    eps = min(1.0, line.length / 100)
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(along + eps, line.length)))
    behind = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(along - eps, 0)))
    tangent = ahead - behind
    foot = shapely.get_coordinates(shapely.line_interpolate_point(line, along))
    side = np.sign(tangent[:, 0] * (xy[:, 1] - foot[:, 1]) - tangent[:, 1] * (xy[:, 0] - foot[:, 0]))
    return rows.assign(along=along, across=side * rows["distance"].to_numpy())


def expression(adata, genes, point_index) -> pd.DataFrame:
    """Dense normalized expression for ``genes`` at the given row positions."""
    import scipy.sparse as sp

    x = adata[point_index, list(genes)].X
    return pd.DataFrame(x.toarray() if sp.issparse(x) else np.asarray(x), columns=list(genes))


def gradient_genes(adata, coords: pd.DataFrame, n: int = 4, min_expressing: float = 0.1) -> list[str]:
    """Genes most correlated (Spearman) with position along the line: half rising, half falling."""
    import scipy.sparse as sp

    x = adata[coords["point_index"].to_numpy()].X
    x = x.tocsc() if sp.issparse(x) else np.asarray(x)
    frac = np.asarray((x > 0).mean(axis=0)).ravel()
    keep = np.flatnonzero(frac >= min_expressing)
    if keep.size == 0:
        return []
    dense = x[:, keep].toarray() if sp.issparse(x) else x[:, keep]
    ranks = pd.DataFrame(dense).rank().to_numpy(copy=True)
    r_along = pd.Series(coords["along"].to_numpy()).rank().to_numpy(copy=True)
    ranks -= ranks.mean(axis=0)
    r_along -= r_along.mean()
    denom = np.sqrt((ranks**2).sum(axis=0) * (r_along**2).sum())
    rho = np.divide(ranks.T @ r_along, denom, out=np.zeros(keep.size), where=denom > 0)
    order = np.argsort(rho)
    names = np.asarray(adata.var_names)[keep]
    rising, falling = names[order[::-1][: n - n // 2]], names[order[: n // 2]]
    return [*rising, *falling]


def gradient_plot(adata, coords: pd.DataFrame, genes: list[str], *, along_bin=25.0, across_bin=25.0):
    """Mean expression per bin along the line and across it (perpendicular, signed)."""
    expr = expression(adata, genes, coords["point_index"].to_numpy())
    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", sharey=True)
    for ax, col, width, label in [
        (ax_a, "along", along_bin, "position along line (µm from start)"),
        (ax_p, "across", across_bin, "signed distance from line (µm, + = left)"),
    ]:
        b = (np.floor(coords[col].to_numpy() / width) + 0.5) * width
        means = expr.groupby(b).mean()
        n = pd.Series(b).value_counts().reindex(means.index)
        means = means[n.to_numpy() >= 10]
        for g in genes:
            ax.plot(means.index, means[g], marker="o", ms=3, label=g)
        ax.set(xlabel=label)
    ax_a.set(ylabel="mean log-normalized expression", title="Along the line")
    ax_p.set(title="Across the line")
    ax_p.axvline(0, color="0.5", lw=1, ls="--")
    ax_p.legend(fontsize=10, loc="best")
    plt.close(fig)
    return fig
