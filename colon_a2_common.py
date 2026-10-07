"""Shared Glasgow colon A2 data load and annotation helpers.

Used by ``colon_a2.py`` (index) and the five beat notebooks. Assumes
``data/colon_a2.sdata.zarr`` exists (see ``build_colon_a2.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from milume.categories import default_categorical_palette

ROOT = Path(__file__).resolve().parent
SDATA_PATH = ROOT / "data" / "colon_a2.sdata.zarr"
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
