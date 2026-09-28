"""Build the Glasgow colon A2 SpatialData for ``colon_a2.py``.

Reads the colon A2 region's Pyxa output with spatialdata-io's ``pyxa`` reader,
from its required counts and cell metadata plus the optional Pyxa Studio export
(``Cluster`` + 3D UMAP). Transcripts are skipped (7.7 GB; the notebook does not
use them).

``--download`` fetches the region from the ``colon/`` folder of the
`Stellaromics/demo <https://huggingface.co/datasets/Stellaromics/demo>`_
dataset into ``data/hf/colon/`` (about 22 GB without the transcripts, plus
12 GB for the unzipped mosaic) and builds from there. ``--source`` takes either
that flat layout or a Pyxa ``Analysis Group`` directory (``ag_output/``,
``Region/mosaic/``).

Writes ``data/colon_a2.sdata.zarr`` (``data/`` is not committed) with:

- ``tables/rna`` — the cells Pyxa Studio kept: raw counts (sparse), ``Cluster``,
  ``obsm["spatial"]`` (x, y, z in um, Pyxa global frame) and ``obsm["X_umap"]``.
  It annotates ``cell_labels`` through the integer ``label_id`` (the ``N`` of
  ``Region_N``).
- ``labels/cell_labels`` — the segmentation polygons rasterized onto the 3D
  mosaic's level-0 voxel grid (``uint32``, 0 = background), with the mosaic's
  transform and pyramid shapes, so the cube cuts the same window from both.
- ``images/mosaic`` — Meteor's 3D nuclear stain (``Region/mosaic/mosaic_3d.ome.zarr``,
  ``c, z, y, x``, ``uint8``), copied level by level (no pyramid recomputed) with
  the same level-0 transform as the labels, so ``LandmarksWidget(sdata)`` finds
  image and labels on one grid.

``SpatialData.write_element`` (spatialdata 0.8) takes no storage options, so the
mosaic is written with plain ``(1, 32, 256, 256)`` chunks, not sharded: a 500 um
inspect window still reads only the chunks it covers.

    uv run python build_colon_a2.py --download --overwrite
    uv run python build_colon_a2.py --overwrite --no-mosaic   # table + labels only
    uv run python build_colon_a2.py --overwrite --no-labels --no-mosaic   # table only, ~30 s
    uv run python build_colon_a2.py --only-mosaic   # add the mosaic to an existing build
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
import zarr
from PIL import Image, ImageDraw
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image3DModel, Labels3DModel, TableModel
from spatialdata.transformations import Scale, Sequence, Translation, set_transformation
from spatialdata_io.experimental import pyxa
from xarray import DataArray, Dataset, DataTree

DATA = Path(__file__).resolve().parent / "data"
HF_REPO = "Stellaromics/demo"
HF_FOLDER = "colon"
DEFAULT_SOURCE = DATA / "hf" / HF_FOLDER
OUT = DATA / "colon_a2.sdata.zarr"
MOSAIC_3D = "mosaic_3d.ome.zarr"
GEOMETRIES = "segmentation_geometries_v1.parquet"
# What the build reads from the Hub; the transcripts (cell_assigned_gene_v1.csv) stay there.
HF_FILES = [
    "cell_by_gene_v1.csv",
    "cell_metadata_v1.csv",
    "pyxa_studio_v1.csv",
    GEOMETRIES,
    f"{MOSAIC_3D}.zip",
]
LABELS = "cell_labels"
MOSAIC = "mosaic"
CELL_PREFIX = "Region_"

# Store chunks match the mosaic's viewer chunks, so a 100 um window reads a few MB.
STORE_CHUNKS = (32, 256, 256)
MOSAIC_CHUNKS = (1, *STORE_CHUNKS)
# One rasterization task: a whole number of store chunks, so tasks never share one.
TILE = (32, 1024, 1024)
# Polygons are traced on the ~0.11 um pixel grid; at the ~0.45 um mosaic voxel their
# staircase vertices add nothing, so they are simplified to a quarter voxel first.
SIMPLIFY_VOXELS = 0.25


def download(dest: Path = DEFAULT_SOURCE) -> Path:
    """Fetch the colon region from the Hub into ``dest`` and unzip its mosaic."""
    import zipfile

    from huggingface_hub import snapshot_download

    root = dest.parent
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        allow_patterns=[f"{HF_FOLDER}/{f}" for f in HF_FILES],
        local_dir=root,
    )
    source = root / HF_FOLDER
    if not (source / MOSAIC_3D).exists():
        t0 = time.perf_counter()
        zipfile.ZipFile(source / f"{MOSAIC_3D}.zip").extractall(source)
        print(f"unzipped {MOSAIC_3D} in {time.perf_counter() - t0:.0f} s", flush=True)
    return source


def pyxa_paths(source: Path) -> tuple[Path, Path, Path]:
    """Tables dir, 3D mosaic and segmentation geometries of a Pyxa output.

    Either a Pyxa ``Analysis Group`` (``ag_output/``, ``Region/mosaic/``) or the
    flat layout of the Hub's ``Stellaromics/demo`` folders.
    """
    if (source / "ag_output").is_dir():
        return source / "ag_output", source / "Region" / "mosaic" / MOSAIC_3D, source / "ag_output" / GEOMETRIES
    return source, source / MOSAIC_3D, source / GEOMETRIES


def mosaic_grid(mosaic: Path) -> dict[str, Any]:
    """Level shapes (z, y, x) and the level-0 voxel frame of Meteor's 3D mosaic."""
    group = zarr.open_group(str(mosaic), mode="r")
    multiscale = group.attrs["ome"]["multiscales"][0]
    axes = [a["name"] for a in multiscale["axes"]]
    zyx = [axes.index(a) for a in ("z", "y", "x")]
    datasets = multiscale["datasets"]
    shapes = [tuple(int(group[d["path"]].shape[i]) for i in zyx) for d in datasets]
    transforms = {t["type"]: t for t in datasets[0]["coordinateTransformations"]}
    return {
        "shapes": shapes,
        "scale": [float(transforms["scale"]["scale"][i]) for i in zyx],
        "translation": [float(transforms["translation"]["translation"][i]) for i in zyx],
    }


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Indices of the concatenated slices ``[s, s + n)`` for each (s, n), in order."""
    total = int(lengths.sum())
    new_starts = np.cumsum(lengths) - lengths
    return np.arange(total) - np.repeat(new_starts, lengths) + np.repeat(starts, lengths)


def _rings_from_row_group(
    geometries: str, row_group: int, grid: dict[str, Any], xy_um: float, z_um: float
) -> dict[str, np.ndarray]:
    """One parquet row group's polygon rings in level-0 voxel coordinates."""
    sz, sy, sx = grid["scale"]
    tz, ty, tx = grid["translation"]
    nz = grid["shapes"][0][0]
    table = pq.ParquetFile(geometries).read_row_group(row_group, columns=["cell_id", "ZIndex", "geometry"])
    cell_id = table.column("cell_id")
    if not pc.all(pc.starts_with(cell_id, CELL_PREFIX)).as_py():
        raise ValueError(f"{Path(geometries).name}: expected every cell_id to be {CELL_PREFIX}<N>")
    label = pc.cast(pc.utf8_slice_codeunits(cell_id, len(CELL_PREFIX)), pa.uint32()).to_numpy()
    zindex = table.column("ZIndex").to_numpy()
    geoms = shapely.from_wkb(table.column("geometry").to_numpy())
    parts, part_of = shapely.get_parts(geoms, return_index=True)
    rings = shapely.get_exterior_ring(parts)
    rings = shapely.transform(
        rings,
        lambda c: np.column_stack(((c[:, 0] * xy_um - tx) / sx + 0.5, (c[:, 1] * xy_um - ty) / sy + 0.5)),
    )
    rings = shapely.simplify(rings, SIMPLIFY_VOXELS)
    # Plane k holds Z_um = (ZIndex + 0.5) * z_um, the reader's plane centre.
    plane = np.rint(((zindex[part_of] + 0.5) * z_um - tz) / sz).astype(np.int32)
    keep = (plane >= 0) & (plane < nz) & ~shapely.is_empty(rings)
    rings, plane, part_label = rings[keep], plane[keep], label[part_of][keep]
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    return {
        "label": part_label,
        "plane": plane,
        "length": np.bincount(ring_of, minlength=len(rings)),
        "coords": coords.astype(np.float32),
        "bounds": shapely.bounds(rings).astype(np.float32),
    }


def load_rings(
    geometries: Path, grid: dict[str, Any], xy_um: float, z_um: float, workers: int
) -> dict[str, np.ndarray]:
    """Every polygon's exterior ring in level-0 voxel coordinates, with its label and plane.

    Row groups are decoded in parallel. Coordinates are shifted by half a voxel so
    PIL's pixel cells line up with the mosaic's voxel centres. Holes are ignored
    (the exterior is filled).
    """
    n_groups = pq.ParquetFile(geometries).metadata.num_row_groups
    with ProcessPoolExecutor(max_workers=max(1, min(workers, n_groups, 60))) as pool:
        parts = list(
            pool.map(
                _rings_from_row_group,
                [str(geometries)] * n_groups,
                range(n_groups),
                [grid] * n_groups,
                [xy_um] * n_groups,
                [z_um] * n_groups,
            )
        )
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def plan_tiles(rings: dict[str, np.ndarray], shape0: tuple[int, int, int]) -> list[dict[str, Any]]:
    """Group rings by rasterization tile (a ring spanning tiles goes to each)."""
    nz, ny, nx = shape0
    tz, ty, tx = TILE
    n_ty, n_tx = -(-ny // ty), -(-nx // tx)
    bounds = rings["bounds"]
    y_lo = np.clip(np.floor(bounds[:, 1] / ty), 0, n_ty - 1).astype(np.int64)
    y_hi = np.clip(np.floor(bounds[:, 3] / ty), 0, n_ty - 1).astype(np.int64)
    x_lo = np.clip(np.floor(bounds[:, 0] / tx), 0, n_tx - 1).astype(np.int64)
    x_hi = np.clip(np.floor(bounds[:, 2] / tx), 0, n_tx - 1).astype(np.int64)
    ring_idx, keys = [], []
    for dy in range(int((y_hi - y_lo).max()) + 1):
        for dx in range(int((x_hi - x_lo).max()) + 1):
            hit = (y_lo + dy <= y_hi) & (x_lo + dx <= x_hi)
            idx = np.flatnonzero(hit)
            ring_idx.append(idx)
            keys.append(((rings["plane"][idx] // tz) * n_ty + y_lo[idx] + dy) * n_tx + x_lo[idx] + dx)
    ring_idx, keys = np.concatenate(ring_idx), np.concatenate(keys)
    order = np.lexsort((rings["label"][ring_idx], keys))
    ring_idx, keys = ring_idx[order], keys[order]

    starts = np.cumsum(rings["length"]) - rings["length"]
    coords = rings["coords"][_ragged_gather(starts[ring_idx], rings["length"][ring_idx])]
    lengths = rings["length"][ring_idx]
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    tasks = []
    edges = np.flatnonzero(np.diff(keys)) + 1
    for lo, hi in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(keys)]]), strict=True):
        key = int(keys[lo])
        kz, rest = divmod(key, n_ty * n_tx)
        ky, kx = divmod(rest, n_tx)
        origin = (kz * tz, ky * ty, kx * tx)
        shape = (min(tz, nz - origin[0]), min(ty, ny - origin[1]), min(tx, nx - origin[2]))
        c0, c1 = int(offsets[lo]), int(offsets[hi])
        tasks.append(
            {
                "origin": origin,
                "shape": shape,
                "label": rings["label"][ring_idx[lo:hi]],
                "plane": rings["plane"][ring_idx[lo:hi]] - origin[0],
                "offsets": offsets[lo : hi + 1] - c0,
                "coords": coords[c0:c1] - np.array([origin[2], origin[1]], dtype=np.float32),
            }
        )
    return tasks


def rasterize_tile(store: str, task: dict[str, Any]) -> int:
    """Fill one tile's polygons plane by plane and write it; later labels win overlaps."""
    depth, height, width = task["shape"]
    block = np.zeros(task["shape"], dtype=np.uint32)
    offsets, coords = task["offsets"], task["coords"]
    for plane in np.unique(task["plane"]):
        image = Image.new("I", (width, height))
        draw = ImageDraw.Draw(image)
        for r in np.flatnonzero(task["plane"] == plane):
            points = coords[offsets[r] : offsets[r + 1]]
            if len(points) >= 3:
                draw.polygon(points.ravel().tolist(), fill=int(task["label"][r]))
        block[plane] = np.asarray(image, dtype=np.int32)
    z0, y0, x0 = task["origin"]
    zarr.open_array(store, mode="r+")[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width] = block
    return len(task["label"])


def rasterize_labels(geometries: Path, grid: dict[str, Any], xy_um: float, z_um: float, store: Path, workers: int) -> None:
    t0 = time.perf_counter()
    rings = load_rings(geometries, grid, xy_um, z_um, workers)
    print(f"read {len(rings['label']):,} polygon rings in {time.perf_counter() - t0:.0f} s", flush=True)
    tasks = plan_tiles(rings, grid["shapes"][0])
    del rings
    shutil.rmtree(store, ignore_errors=True)
    zarr.create_array(
        store=str(store), shape=grid["shapes"][0], chunks=STORE_CHUNKS, dtype="uint32", fill_value=0
    )
    t1 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=min(workers, 60)) as pool:
        for n in pool.map(rasterize_tile, [str(store)] * len(tasks), tasks, chunksize=4):
            done += n
    print(f"rasterized {done:,} rings in {len(tasks)} tiles in {time.perf_counter() - t1:.0f} s", flush=True)


def labels_element(store: Path, grid: dict[str, Any]) -> DataTree:
    """Multiscale labels on the mosaic's grid: level 0 plus strided (nearest) levels."""
    level0 = da.from_zarr(str(store))
    n0 = grid["shapes"][0]
    levels = {}
    for i, shape in enumerate(grid["shapes"]):
        step = [max(1, round(a / b)) for a, b in zip(n0, shape, strict=True)]
        array = level0[:: step[0], :: step[1], :: step[2]][: shape[0], : shape[1], : shape[2]]
        array = array.rechunk(tuple(min(c, s) for c, s in zip(STORE_CHUNKS, shape, strict=True)))
        # Coordinates are pixel centres in level-0 units, as spatialdata assigns them.
        coords = {ax: np.linspace(0, a, b + 1)[:-1] + a / b / 2 for ax, a, b in zip("zyx", n0, shape, strict=True)}
        levels[f"scale{i}"] = Dataset({"image": DataArray(array, dims=("z", "y", "x"), coords=coords)})
    tree = DataTree.from_dict(levels)
    transform = Sequence(
        [Scale(grid["scale"], axes=("z", "y", "x")), Translation(grid["translation"], axes=("z", "y", "x"))]
    )
    set_transformation(tree, {"global": transform}, set_all=True)
    Labels3DModel.validate(tree)
    return tree


def mosaic_element(source: Path) -> DataTree:
    """Meteor's 3D mosaic as a multiscale image on the labels' grid, levels as stored."""
    _, mosaic, _ = pyxa_paths(source)
    grid = mosaic_grid(mosaic)
    group = zarr.open_group(str(mosaic), mode="r")
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    # Read whole shards per task (then split into store chunks); drop t.
    raw = [group[d["path"]] for d in datasets]
    levels = [da.from_zarr(a, chunks=a.shards or a.chunks)[0] for a in raw]
    transform = Sequence(
        [Scale(grid["scale"], axes=("z", "y", "x")), Translation(grid["translation"], axes=("z", "y", "x"))]
    )
    image0 = Image3DModel.parse(
        levels[0],
        dims=("c", "z", "y", "x"),
        scale_factors=None,
        chunks=MOSAIC_CHUNKS,
        transformations={"global": transform},
    )
    n0 = grid["shapes"][0]
    tree = {"scale0": Dataset({"image": image0})}
    for i, level in enumerate(levels[1:], start=1):
        shape = level.shape[1:]
        array = level.rechunk(tuple(min(c, s) for c, s in zip(MOSAIC_CHUNKS, level.shape, strict=True)))
        # Coordinates are pixel centres in level-0 units, as spatialdata assigns them.
        coords = {ax: np.linspace(0, a, b + 1)[:-1] + a / b / 2 for ax, a, b in zip("zyx", n0, shape, strict=True)}
        coords["c"] = image0.coords["c"].values
        tree[f"scale{i}"] = Dataset({"image": DataArray(array, dims=("c", "z", "y", "x"), coords=coords)})
    tree = DataTree.from_dict(tree)
    set_transformation(tree, {"global": transform}, set_all=True)
    Image3DModel.validate(tree)
    return tree


def write_mosaic(sdata: SpatialData, source: Path, *, overwrite: bool) -> None:
    """Add ``images/mosaic`` to the store backing ``sdata``."""
    t0 = time.perf_counter()
    # Replace any copy read from the store first, so none is backed by the path deleted below.
    sdata.images[MOSAIC] = mosaic_element(source)
    if f"images/{MOSAIC}" in sdata.elements_paths_on_disk():
        if not overwrite:
            raise SystemExit(f"images/{MOSAIC} is already in {sdata.path}: pass --overwrite to replace it")
        # write_element cannot overwrite inside its own store.
        sdata.delete_element_from_disk(MOSAIC)
    sdata.write_element(MOSAIC)
    print(f"wrote images/{MOSAIC} in {time.perf_counter() - t0:.0f} s", flush=True)


def build(source: Path, *, labels: bool, workers: int, scratch: Path) -> SpatialData:
    """Cells Pyxa Studio kept, with their clusters, and optionally their 3D labels."""
    tables, mosaic, geometries = pyxa_paths(source)
    sdata = pyxa(
        tables,
        cell_assigned_gene=False,
        segmentation_geometries=False,
        pyxa_studio=True,
    )
    table = sdata["rna"]
    # Studio drops cells that fail Pyxa's filters; they carry no cluster.
    table = table[table.obs["Cluster"].notna()].copy()
    table.obs["Cluster"] = table.obs["Cluster"].cat.remove_unused_categories()
    table.X = sparse.csr_matrix(table.X, dtype=np.float32)
    table.uns["pyxa"] = {"analysis_group": str(source)}
    if not labels:
        return SpatialData(tables={"rna": TableModel.parse(table)})

    # Pixel and plane sizes of the polygons, from the cells' centroids in both units.
    xy_um = float(np.median(table.obsm["spatial"][:, 0] / table.obs["X_pixels"]))
    z_um = float(np.median(table.obsm["spatial"][:, 2] / table.obs["Z_pixels"]))
    grid = mosaic_grid(mosaic)
    rasterize_labels(geometries, grid, xy_um, z_um, scratch, workers)

    table.obs["label_id"] = table.obs_names.str.slice(len(CELL_PREFIX)).astype(np.int64)
    table.obs["region"] = LABELS
    table.obs["region"] = table.obs["region"].astype("category")
    table = TableModel.parse(table, region=LABELS, region_key="region", instance_key="label_id")
    return SpatialData(labels={LABELS: labels_element(scratch, grid)}, tables={"rna": table})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Pyxa output: an 'Analysis Group' dir or a flat Hub folder (default: data/hf/colon)",
    )
    parser.add_argument(
        "--download", action="store_true", help=f"fetch {HF_REPO}/{HF_FOLDER} into --source first"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-labels", dest="labels", action="store_false", help="skip the 3D labels")
    parser.add_argument("--no-mosaic", dest="mosaic", action="store_false", help="skip the 3D image")
    parser.add_argument(
        "--only-mosaic", action="store_true", help="write only images/mosaic into the existing --out store"
    )
    # Windows caps a process pool at 61 workers.
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 60))
    args = parser.parse_args()
    if args.download:
        args.source = download(args.source)

    if args.only_mosaic:
        sdata = read_zarr(args.out)
        write_mosaic(sdata, args.source, overwrite=args.overwrite)
        # Builds before the mosaic was in the SpatialData pointed the notebook at Meteor's
        # store. write_element cannot overwrite the table in place, so drop that one key.
        if "mosaic_3d" in sdata.tables["rna"].uns.get("pyxa", {}):
            del zarr.open_group(str(args.out / "tables" / "rna" / "uns" / "pyxa"), mode="r+")["mosaic_3d"]
            sdata.write_consolidated_metadata()
        print(json.dumps({"out": str(args.out), "images": list(sdata.images), "labels": list(sdata.labels)}, indent=1))
        return

    scratch = args.out.parent / "_labels_level0.zarr"
    sdata = build(args.source, labels=args.labels, workers=args.workers, scratch=scratch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sdata.write(args.out, overwrite=args.overwrite)
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f} s", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    if args.mosaic:
        write_mosaic(sdata, args.source, overwrite=args.overwrite)
    table = sdata.tables["rna"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_obs": table.n_obs,
                "n_vars": table.n_vars,
                "labels": list(sdata.labels),
                "images": list(sdata.images),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
