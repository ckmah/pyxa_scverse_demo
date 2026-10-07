"""Glasgow colon A2: index for the why-3D beat notebooks.

Loads the SpatialData that ``build_colon_a2.py`` builds into ``data/`` (not
committed) from the ``colon/`` folder of the Stellaromics/demo dataset on Hugging Face.
Five short notebooks each contrast what a flat 2D view implies with what 3D inspection
and measurement show. Shared load and annotation logic lives in ``colon_a2_common.py``.

    uv run python build_colon_a2.py --download --overwrite
    uv run marimo edit colon_a2.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from colon_a2_common import DOMAIN_LEVEL, SDATA_PATH, apply_mpl_theme

    return DOMAIN_LEVEL, SDATA_PATH, apply_mpl_theme, mo


@app.cell
def _(apply_mpl_theme, mo):
    apply_mpl_theme(mo.app_meta().theme)
    return


@app.cell(hide_code=True)
def _(DOMAIN_LEVEL, SDATA_PATH, mo):
    mo.md(f"""
    # Why 3D matters: Glasgow colon A2

    A 1020-plex spatial transcriptomics section (~5 mm × 4 mm × 140 µm) from colorectal
    cancer tissue on the Stellaromics Pyxa platform. For tool builders, cell biologists,
    pathologists, biomedical researchers, and platform folks: five short notebooks each
    contrast what a **flat 2D map implies** with what **3D inspection and measurement** show.

    Data: `{SDATA_PATH.name}` (build with `build_colon_a2.py`). Annotations and marker genes
    load from `colon_a2_common.py` in every beat notebook.

    | Beat | Flatten → collapse | Notebook |
    |------|-------------------|----------|
    | 1 | Crypt **rings** on the map → **tubes through Z** (Inspect + orbit) | [`colon_a2_beat1_rings.py`](colon_a2_beat1_rings.py) |
    | 2 | Broken **arcs** on one plane → **continuous walls** through depth (Cross-section) | [`colon_a2_beat2_cross_section.py`](colon_a2_beat2_cross_section.py) |
    | 3 | Flat-map **neighbors** → microns apart in **Z** (`nearest_distances`) | [`colon_a2_beat3_neighbors.py`](colon_a2_beat3_neighbors.py) |
    | 4 | One flat composition mix → **stacked niches** by z bin | [`colon_a2_beat4_niches.py`](colon_a2_beat4_niches.py) |
    | 5 | Prettier viewing → **pick the right cells** for DE | [`colon_a2_beat5_analyze.py`](colon_a2_beat5_analyze.py) |

    ```bash
    uv run marimo edit colon_a2_beat1_rings.py   # start with beat 1
    ```

    /// admonition | Acknowledgements
    **School of Cancer Sciences, University of Glasgow, UK**: Marta Campillo Poveda, Anthony Chalmers, Yoana Doncheva, Joanne Edwards, Andrea Gonzalez Ciscar, **Nigel Jamieson**, Claire Kennedy Dietrich, Ghazal Latif, Assya Legrini, Josefina Marinez Vasquez, Pamela McCall, Mari-Claire McGuigan, Luke McNickle, Tengyu Zhang

    **University of Edinburgh, UK**: Gerry Thompson

    **Stellaromics Inc, Boston, MA, USA**: Leah Carlson, Jeremy Lambert, Clarence Mah, Raghav Padmanabhan, Chan Park, Daphne Sze, Alexis Wong
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
