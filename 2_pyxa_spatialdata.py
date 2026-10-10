# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.0",
#     "milume>=1.3.1",
#     "scanpy",
#     "igraph",
#     # pyxa reader: ckmah/spatialdata-io, pyxa-reader branch, pinned to its head commit
#     "spatialdata-io @ git+https://github.com/ckmah/spatialdata-io@9a85304fc2f11f04e187f161c399dfb3a7900b3c",
#     "spatialdata",
#     "fsspec",
#     "huggingface-hub",
#     "zarr",
#     "polars",
#     "geopandas",
#     "shapely",
#     "matplotlib",
#     "anywidget",
#     "traitlets",
# ]
# ///
"""Pyxa to SpatialData: read the xsmall crop with the pyxa reader and browse it by z-plane.

    uv run marimo edit 2_pyxa_spatialdata.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import fsspec
    import marimo as mo
    import matplotlib.pyplot as plt
    from spatialdata import get_extent
    import polars as pl
    import geopandas as gpd
    import zarr

    from spatialdata_io.experimental import pyxa

    DATA_DIR = Path(__file__).resolve().parent / "data"
    return DATA_DIR, Path, fsspec, get_extent, mo, pl, plt, pyxa


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🔬 Pyxa 3D spatial omics → SpatialData

    Reads the `xsmall` crop (a 100 × 100 × 100 µm cube, 187 cells) of the
    [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) dataset with the experimental
    `pyxa` reader from [spatialdata-io](https://github.com/scverse/spatialdata-io), writes it to SpatialData
    Zarr, and browses it one z-plane at a time.

    ## 1. Download the data

    Downloads `xsmall/` from the Hugging Face Hub through fsspec's `hf://` filesystem (provided by
    `huggingface_hub`).
    """)
    return


@app.cell
def _(DATA_DIR, Path, fsspec):
    xsmall_dir = DATA_DIR / "xsmall"
    xsmall_dir.mkdir(parents=True, exist_ok=True)

    fs = fsspec.filesystem("hf")
    for remote_path in fs.ls(
        "datasets/Stellaromics/demo/xsmall", detail=False
    ):
        local_path = xsmall_dir / Path(remote_path).name
        if not local_path.exists():
            fs.get(remote_path, str(local_path))

    sorted(p.name for p in xsmall_dir.iterdir())
    return (xsmall_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
    ## 2. Convert to SpatialData

    The reader returns every element in µm in the `global` coordinate system, then writes SpatialData Zarr.
    /// admonition | Tip\n
    The `pyxa()` reader will read the zipped OME-Zarr DAPI mosaic in place. No need to unzip.
    ///
    """),
            mo.mermaid("""
    %%{init: {"theme": "base", "themeVariables": {"lineColor": "#888888", "textColor": "#888888", "edgeLabelBackground": "transparent", "clusterBkg": "transparent", "clusterBorder": "#888888"}, "flowchart": {"nodeSpacing": 25, "rankSpacing": 90, "curve": "basis"}}}%%
    flowchart LR
        subgraph IN["Input files"]
            direction TB
            A1(mosaic_3d.ome.zarr.zip)
            A2(cell_assigned_gene_v1.csv)
            A3(segmentation_geometries_v1.parquet)
            A5(cell_metadata_v1.csv)
            A4(cell_by_gene_v1.csv)
        end
        subgraph OUT["SpatialData"]
            direction TB
            B1(images/mosaic_image)
            B2(points/transcripts)
            B3(shapes/cell_boundaries_z)
            B4(shapes/cell_boundaries)
            B5(tables/rna)
        end
        A1 --> B1
        A2 --> B2
        A3 --> B3
        A3 --> B4
        A5 --> B5
        A4 --> B5
        classDef file fill:#6b7280,stroke:#9ca3af,color:#ffffff
        classDef img fill:#2563eb,stroke:#60a5fa,color:#ffffff
        classDef pts fill:#d97706,stroke:#fbbf24,color:#ffffff
        classDef shp fill:#16a34a,stroke:#4ade80,color:#ffffff
        classDef tbl fill:#db2777,stroke:#f472b6,color:#ffffff
        style IN fill:none,stroke:#888888,color:#888888
        style OUT fill:none,stroke:#888888,color:#888888
        class A1,A2,A3,A4,A5 file
        class B1 img
        class B2 pts
        class B3,B4 shp
        class B5 tbl
    """),
        ]
    )
    return


@app.cell
def _(pl, xsmall_dir):
    pl.read_csv(xsmall_dir / "cell_assigned_gene_v1.csv")
    return


@app.cell
def _(pl, xsmall_dir):
    pl.read_csv(xsmall_dir / "cell_by_gene_v1.csv")
    return


@app.cell
def _(xsmall_dir):
    from spatialdata_io.readers.pyxa import (
        _get_image,
    )  # internal function for demo purposes only; use pyxa

    _get_image(xsmall_dir / "mosaic_3d.ome.zarr.zip")
    return


@app.cell
def _(pl, xsmall_dir):
    pl.read_csv(xsmall_dir / "cell_metadata_v1.csv")
    return


@app.cell
def _(DATA_DIR, pyxa, xsmall_dir):
    sdata = pyxa(xsmall_dir)
    sdata.write(DATA_DIR / "xsmall.zarr", overwrite=True)
    sdata
    return (sdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Explore in the Milume widget

    `milume.peek` opens the SpatialData object in an interactive 3D viewer: pan and zoom the DAPI stack, draw
    landmarks, and open the Inspect view on any region. **Note:** `xsmall` is a small mouse brain subset used here
    only to walk through the tools; it says nothing about colon biology (see `colon_a2.py` for that).
    """)
    return


@app.cell
def _(sdata):
    import scanpy as sc

    # Quick clustering so the widget has a category to colour by (187 cells x 241 genes).
    _adata = sdata.tables["rna"]
    _counts = _adata.X.copy()
    sc.pp.normalize_total(_adata)
    sc.pp.log1p(_adata)
    sc.pp.pca(_adata, n_comps=20, random_state=0)
    sc.pp.neighbors(_adata, random_state=0)
    sc.tl.leiden(_adata, flavor="igraph", n_iterations=2, resolution=0.8, random_state=0)
    _adata.obs["leiden"] = _adata.obs["leiden"].astype("category")
    _adata.layers["counts"] = _counts
    clustered = sdata
    return (clustered,)


@app.cell(expand_output=True)
def _(clustered, mo):
    import milume

    peek = mo.ui.anywidget(milume.peek(clustered, color="leiden"))
    peek
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Browse data in 3D

    The full-resolution DAPI stack is loaded into memory, and transcripts and z-plane polygons are grouped by the
    image plane their z falls in, so each slider step only looks up one plane. Transcripts are blue if assigned to
    a cell and orange if not.
    """)
    return


@app.cell
def _(get_extent, sdata):
    image = sdata["mosaic_image"]["scale0"]["image"]
    stack = image.isel(c=0).values  # (z, y, x) DAPI planes
    extent = get_extent(image)
    z_min, z_max = extent["z"]
    dz = (z_max - z_min) / len(stack)

    transcripts = sdata["transcripts"].compute()
    polygons = sdata["cell_boundaries_z"]
    return dz, extent, polygons, stack, transcripts, z_max, z_min


@app.cell
def _(dz, extent, plt, polygons, stack):
    import io

    STEP_UM = 2  # render every 5 µm
    x_lo, x_hi = extent["x"]
    y_lo, y_hi = extent["y"]
    px = (x_hi - x_lo) / stack.shape[2]  # µm per pixel
    py = (y_hi - y_lo) / stack.shape[1]
    z_idx = range(
        0, stack.shape[0], round(STEP_UM / dz)
    )  # slice indices to render
    y_idx = range(0, stack.shape[1], round(STEP_UM / py))
    x_idx = range(0, stack.shape[2], round(STEP_UM / px))

    TX_COLORS = {
        True: "#2a78d6",
        False: "#eb6834",
    }  # assigned / unassigned transcripts
    colors = {
        c: plt.cm.tab20(i % 20)
        for i, c in enumerate(polygons["cell_id"].unique())
    }

    def render(
        img, extent, labels, title, tx, cols, polys=None, cells=(), ylim=None
    ):
        """One frame as PNG bytes: image + cells + transcripts."""
        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
        ax.imshow(img, cmap="gray", extent=extent)
        if polys is not None:
            polys.plot(
                ax=ax,
                color=[colors[c] for c in polys["cell_id"]],
                alpha=0.3,
                linewidth=0,
            )
        for lo, hi, z, cid in cells:
            ax.fill_between(
                [lo, hi],
                z - dz / 2,
                z + dz / 2,
                color=colors[cid],
                alpha=0.3,
                linewidth=0,
            )
        ax.scatter(
            tx[cols[0]], tx[cols[1]], s=6, c=tx["assigned"].map(TX_COLORS)
        )
        ax.set(
            xlim=extent[:2],
            ylim=ylim or extent[2:],
            xlabel=labels[0],
            ylabel=labels[1],
            title=title,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    return py, render, x_hi, x_lo, y_hi, y_idx, y_lo, z_idx


@app.cell
def _(
    dz,
    mo,
    polygons,
    render,
    stack,
    transcripts,
    x_hi,
    x_lo,
    y_hi,
    y_lo,
    z_idx,
    z_min,
):
    def xy(k):
        z = z_min + (k + 0.5) * dz
        in_plane = lambda df, col: df[(df[col] - z_min) // dz == k]
        return render(
            stack[k],
            (x_lo, x_hi, y_hi, y_lo),
            ("x (µm)", "y (µm)"),
            f"xy · z = {z:.0f} µm",
            in_plane(transcripts, "z"),
            ("x", "y"),
            polys=in_plane(polygons, "Z_um"),
        )

    frames_xy = [
        xy(k) for k in mo.status.progress_bar(z_idx, title="Rendering xy", remove_on_exit=True)
    ]
    return (frames_xy,)


@app.cell
def _(
    dz,
    mo,
    polygons,
    py,
    render,
    stack,
    transcripts,
    x_hi,
    x_lo,
    y_idx,
    y_lo,
    z_max,
    z_min,
):
    from shapely.geometry import LineString

    def xz(j):
        y = y_lo + (j + 0.5) * py
        cut = polygons.geometry.intersection(
            LineString([(x_lo, y), (x_hi, y)])
        )  # each cell polygon cut by the slice
        cut = cut[~cut.is_empty]
        return render(
            stack[:, j, :],
            (x_lo, x_hi, z_max, z_min),
            ("x (µm)", "z (µm)"),
            f"xz · y = {y:.0f} µm",
            transcripts[(transcripts.y - y).abs() <= 1],
            ("x", "z"),
            cells=zip(
                cut.bounds.minx,
                cut.bounds.maxx,
                polygons.Z_um[cut.index],
                polygons.cell_id[cut.index],
            ),
            ylim=(
                polygons.Z_um.max() + dz / 2,
                polygons.Z_um.min() - dz / 2,
            ),  # show whole cells, even past the image
        )

    frames_xz = [
        xz(j) for j in mo.status.progress_bar(y_idx, title="Rendering xz", remove_on_exit=True)
    ]
    return (frames_xz,)


@app.cell
def _(frames_xy, frames_xz):
    import base64

    import anywidget
    import traitlets


    def _b64(frames):
        return [base64.b64encode(f).decode() for f in frames]


    class SliceViewer(anywidget.AnyWidget):
        """Two slice browsers; frames are sent once and swapped client-side,
        so dragging a slider never re-runs a cell (no flicker)."""

        _esm = """
        function panel(label, frames, start) {
          const box = document.createElement("div");
          const slider = document.createElement("input");
          slider.type = "range"; slider.min = 0; slider.max = frames.length - 1;
          slider.value = start; slider.style.width = "500px";
          const img = document.createElement("img");
          img.width = 500; img.height = 500; img.style.display = "block";
          const imgs = frames.map((b) => {
            const i = new Image(); i.src = "data:image/png;base64," + b; return i;
          });
          const show = () => { img.src = imgs[slider.value].src; };
          slider.oninput = show; show();
          const lab = document.createElement("div");
          lab.textContent = label;
          box.append(lab, slider, img);
          return box;
        }
        export default {
          render({ model, el }) {
            el.style.cssText = "display:flex;gap:16px;flex-wrap:wrap";
            const xy = model.get("frames_xy"), xz = model.get("frames_xz");
            el.append(
              panel("z-plane (5 µm steps)", xy, xy.length >> 1),
              panel("y-slice (5 µm steps)", xz, xz.length >> 1),
            );
          },
        };
        """
        frames_xy = traitlets.List().tag(sync=True)
        frames_xz = traitlets.List().tag(sync=True)


    viewer = SliceViewer(frames_xy=_b64(frames_xy), frames_xz=_b64(frames_xz))
    return (viewer,)


@app.cell
def _(viewer):
    viewer
    return


if __name__ == "__main__":
    app.run()
