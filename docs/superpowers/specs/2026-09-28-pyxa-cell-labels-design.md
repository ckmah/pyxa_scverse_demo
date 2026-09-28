# Pyxa reader: 3D cell labels, mosaic lookup, sparse counts

Date: 2026-09-28. Target: spatialdata-io `pyxa-reader` branch
(scverse/spatialdata-io#425), `src/spatialdata_io/readers/pyxa.py`.

## Goal

Move the colon demo's conversion (`pyxa_scverse_demo/build_colon_a2.py`) into
the `pyxa` reader, so one call returns a SpatialData that
`LandmarksWidget(sdata)` can open with its inspect cube:

```python
sdata = pyxa(path, labels=True, cell_assigned_gene=False)
sdata.write("region.sdata.zarr")
```

The demo's build script then reduces to that call and a write.

## Non-goals

- Filtering cells. Cells Pyxa Studio filtered out stay in the table with
  missing `Cluster` / `X_umap`, as today.
- Labels on a grid other than the mosaic's. No `labels_voxel_size`.
- Auto-detecting the Pyxa `Analysis Group` layout (`ag_output/` next to
  `Region/mosaic/`). There, the mosaic is passed with `image=`.
- Stitching Meteor's per-FOV masks (`Region/segment/MASK*.tif`).
- Filling polygon holes. Exteriors are filled, as in the demo.

## API

```python
pyxa(
    path=None,
    dataset_id="pyxa",
    *,
    cell_by_gene=None,
    cell_metadata=None,
    cell_assigned_gene=None,
    segmentation_geometries=None,
    pyxa_studio=None,
    image=None,
    shapes=None,
    labels=False,
)
```

- `image`: resolved like the other optional inputs. `None` reads
  `path / "mosaic_3d.ome.zarr"`, else `path / "mosaic_3d.ome.zarr.zip"`, when
  present, and skips it otherwise. `True` requires one of them. `False` skips.
  A path reads that directory or `.zip`. `image_path` is removed outright
  (no deprecation; #425 is still a draft with no users). `image` is
  keyword-only.
- `labels`: `True` rasterizes `labels["cell_labels"]` from the segmentation
  polygons onto the mosaic's grid. It needs both the geometries and the image;
  if either is missing it raises
  `ValueError("labels=True needs segmentation_geometries and a mosaic image; missing: ...")`.
- `shapes`: `None` returns the footprint and z-plane shapes whenever the
  geometries are read and `labels` is `False`. `True` returns them (and
  requires the geometries). `False` skips them.
- CLI (`spatialdata_io pyxa`): `--image PATH | --no-image`, `--labels`,
  `--shapes/--no-shapes`, next to the existing input flags.

## Table

- `X` is a CSR matrix with the counts' dtype (dense 492k x 1,020 float64 is
  4 GB).
- Annotation:
  - with labels: `region="cell_labels"`, `region_key="region"`,
    `instance_key="label_id"`;
  - otherwise, with shapes: the footprints, as today (`instance_key="cell_id"`);
  - otherwise: none.
- `obs["cell_id"]` always stays.

### Label ids

- `label_id` is the trailing integer of `cell_id` after any prefix
  (`(\d+)$`): `Region_17` gives 17, `ROI2_17` gives 17.
- It is used only when every cell has one, all are unique, all are > 0 (0 is
  background) and all fit in `uint32`.
- Otherwise cells are numbered 1..n in table order, and the reader logs
  (INFO) which rule was used and why.
- Polygons get their label through `cell_id` joined to the table, so labels
  and table always agree. Polygons whose `cell_id` is not in the table are
  dropped, with a logged count.

## Rasterization (lazy tiles)

1. **Grid.** From the mosaic's OME-NGFF metadata: each level's (z, y, x)
   shape and the level-0 scale and translation (µm). This is factored out of
   `_get_image` so the image and labels share one definition.
2. **Rings.** The parquet is decoded by row group in a thread pool (shapely
   and pyarrow release the GIL) into flat arrays: label, plane, ring lengths,
   float32 vertices and bounds. For each polygon part:
   - take the exterior ring;
   - convert pixel coordinates to µm with the already inferred `xy_size`, then
     to level-0 voxel coordinates, with a +0.5 voxel shift so PIL pixel cells
     line up with voxel centres;
   - simplify to 0.25 voxel;
   - the plane is `round(((ZIndex + 0.5) * z_size - t_z) / s_z)`.

   Rings whose plane is outside the grid, or that are empty after
   simplification, are dropped, with a logged count. This step is eager: for
   the full colon A2 region it is 23.2M rings, 6.3 GB and about 64 s, plus
   about 20 s to plan the tiles (measured in the spike).
3. **Levels.** Level L's step relative to level 0 is `round(n0 / n)` per
   axis (the colon mosaic's coarse levels repeat 17 planes, giving a z step of
   17 at those levels). Level L with step (dz, dy, dx) keeps the rings on planes where `plane % dz == 0`, divides their xy by
   (dx, dy) and their plane by dz. This equals nearest-neighbour striding of
   level 0 without drawing level 0 again. Level shapes are taken from the
   mosaic, so they match it exactly; draws past a level's edge are clipped.
4. **Tiles.** For each level, rings are grouped by 32 x 1024 x 1024 tile
   (clipped to the level's shape). A ring that crosses tiles goes to each of
   them. Rings are sorted by label within a tile, so where rings overlap the
   higher id wins, deterministically.
5. **Array.** Each tile is `dask.delayed(_rasterize_tile)(tile_rings)`: PIL
   `ImageDraw.polygon` fills a mode-`I` image per plane, into a `uint32`
   block. Tiles with no rings are `da.zeros`. Blocks are joined with
   `da.block` and rechunked to 32 x 256 x 256 (clipped).
6. **Element.** The levels become a `DataTree` with pixel-centre coords, as
   in `_get_image`, and the image's `global` transform (`set_all=True`), and
   are validated with `Labels3DModel`.

Nothing is drawn until compute or write. The default threaded scheduler is the
right one: in the spike it was within 20% of a 60-process pool (and gave the
same voxels), while dask's process scheduler was 10x slower because it
pickles every 128 MB tile. For the full colon region, the whole level-0
rasterization is estimated at about 3 minutes on 32 threads.

## Mosaic zip

A `.zip` is opened with `zarr.storage.ZipStore(path, mode="r")`. The group
path inside it is the zip's single top-level entry
(`mosaic_3d.ome.zarr/`). Arrays come from `da.from_zarr` on that store, and
the element keeps a reference to it. The docs note that zip-backed elements
are read with the threaded scheduler (ZipStore is not safe to use across
processes). Writing the SpatialData copies the image out of the zip.

## Errors and logging

- `labels=True` without the geometries or the image: `ValueError` naming what
  is missing.
- `shapes=True` without the geometries: `FileNotFoundError`, as for any
  required input.
- INFO logs: the label-id rule and why; the number of polygons dropped (no
  table cell, off-grid plane, empty); rings, tiles and levels planned.

## Testing (`tests/test_pyxa.py`, public Hub xsmall / small data)

- `_label_ids`: `Region_N`; mixed prefixes; the fallback on collisions, on a
  missing trailing integer and on 0.
- Labels match the mosaic's level shapes and transform (xsmall, small).
- A known xsmall cell: voxels inside its polygon on its plane carry its
  `label_id`; the background is 0; the ids drawn are a subset of the table's
  `label_id`.
- Level L equals level 0 strided by its step.
- Lazy: a patched `_rasterize_tile` is not called until `.compute()`.
- Zip: `image=<.zip>` gives the same image values as the unzipped
  directory.
- The mosaic is found in `path` both unzipped and zipped; `image=False` skips
  it.
- `labels=True` skips shapes by default; `shapes=True` returns both, and the
  table annotates the labels.
- Errors: `labels=True` without the image or geometries.
- `X` is CSR.
- Round trip: write, read, and the table's region / instance key resolve
  against `labels["cell_labels"]`.
- CLI: `--labels`, `--no-shapes`, `--no-image`.
- Manual (not CI): the full colon A2 region, with read and write timings
  recorded in the PR.

## Follow-ups outside the reader

- `pyxa_scverse_demo/build_colon_a2.py` becomes a call to
  `pyxa(source, labels=True, cell_assigned_gene=False)` and a write; the
  `--download` step and the flat Hub layout stay.
- Callers of `image_path=` move to `image=`: the pyxa tests and CLI
  (`__main__.py`) in spatialdata-io, `pyxa_scverse_demo/demo_pyxa.ipynb`, and
  the Stellaromics/demo dataset card's usage snippet (which can then drop its
  unzip step and gain `labels=True`).
