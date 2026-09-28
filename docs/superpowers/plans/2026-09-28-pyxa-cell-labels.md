# Pyxa Reader Cell Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pyxa(path, labels=True)` returns a SpatialData with the mosaic image, 3D `cell_labels` rasterized lazily onto the mosaic's grid and a sparse table annotating them, so the colon demo's build script reduces to one reader call.

**Architecture:** Label rasterization lives in a new focused module `src/spatialdata_io/readers/_pyxa_labels.py` (grid, label ids, ring decoding, tile planning, PIL tile drawing, lazy dask assembly), with no import from `pyxa.py`. `pyxa.py` gains mosaic lookup (directory or zip, read in place), sparse counts, and the `image=` / `shapes=` / `labels=` options that wire the module in. Nothing is drawn until the returned SpatialData is computed or written.

**Tech Stack:** Python 3.11+, spatialdata 0.8, zarr 3 (`ZipStore`), dask (`delayed`, `da.block`), shapely 2, pyarrow, Pillow (`ImageDraw`, already installed via scikit-image), pytest, click.

**Spec:** `docs/superpowers/specs/2026-09-28-pyxa-cell-labels-design.md` (in pyxa_scverse_demo).

## Global Constraints

- Code changes go to `D:\clarence\spatialdata-io`, branch `pyxa-reader` (draft upstream PR scverse/spatialdata-io#425). Tasks 1–7 commit there. Do not push without the user's go-ahead.
- Tasks 8–9 are in `D:\clarence\pyxa_scverse_demo`, branch `colon-a2-demo`.
- Run tests from the spatialdata-io root with its venv: `.venv/Scripts/python.exe -m pytest ...` (Windows). The fixture `data/pyxa_xsmall/` (unzipped `mosaic_3d.ome.zarr`, no zip) is already there.
- Ruff line length 120; follow the file's existing docstring style (numpy-ish, backticked names), `from __future__ import annotations`.
- `image_path` is removed outright (no deprecation). `image` is keyword-only.
- Label id = trailing integer of `cell_id` (`(\d+)$`) when every cell has one, all unique, all > 0 and all < 2**31; otherwise 1..n in table order; log which rule and why.
- Labels grid = the mosaic's grid (level shapes, level-0 scale + translation). `labels=True` without geometries or an image raises `ValueError("labels=True needs segmentation_geometries and a mosaic image; missing: ...")`.
- With labels, shapes are skipped unless `shapes=True`; the table annotates `cell_labels` via `instance_key="label_id"`, `region_key="region"`.
- `X` is CSR with the counts' dtype; no cells filtered.
- Tile = 32 x 1024 x 1024 voxels (clipped); rechunk to 32 x 256 x 256 (clipped). Rings simplified to 0.25 voxel. Higher label id wins overlaps. Level step = `max(1, round(n0 / n))` per axis.
- Default dask threaded scheduler; docs must not recommend the process scheduler.
- End every commit message with `Co-authored-by: ckmah <clarence.mah@stellaromics.com>`.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/spatialdata_io/readers/_pyxa_labels.py` (create) | `_MosaicGrid`, `_label_ids`, `_Rings`, `_read_rings`, `_Tile`, `_plan_tiles`, `_rasterize_tile`, `_labels_level`, `_get_labels` |
| `src/spatialdata_io/readers/pyxa.py` (modify) | `_open_mosaic`, `_mosaic_grid`, `_resolve_image`; `_get_image` accepts dir or zip; sparse `X`; `pyxa(image=, shapes=, labels=)` |
| `src/spatialdata_io/_constants/_constants.py` (modify) | New `PyxaKeys`: `MOSAIC_FILE`, `MOSAIC_ZIP_FILE`, `CELL_LABELS`, `LABEL_ID` |
| `src/spatialdata_io/__main__.py` (modify) | CLI: `--image`, `--no-image`, `--labels`, `--shapes/--no-shapes`; drop `--image-path` |
| `tests/test_pyxa.py` (modify) | Existing tests moved to `image=`; new tests for every unit above |
| `docs/changelog.md`, `README.md` (modify) | Document the new options |

---

### Task 1: Sparse counts and new keys

**Files:**
- Modify: `src/spatialdata_io/_constants/_constants.py` (class `PyxaKeys`, after `UMAP_KEY = "X_umap"`)
- Modify: `src/spatialdata_io/readers/pyxa.py` (`_get_table`, imports)
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Produces: `PyxaKeys.MOSAIC_FILE == "mosaic_3d.ome.zarr"`, `PyxaKeys.MOSAIC_ZIP_FILE == "mosaic_3d.ome.zarr.zip"`, `PyxaKeys.CELL_LABELS == "cell_labels"`, `PyxaKeys.LABEL_ID == "label_id"`. `_get_table(...)` returns AnnData with `scipy.sparse.csr_matrix` `X`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pyxa.py`:

```python
def test_pyxa_keys_labels() -> None:
    assert PyxaKeys.MOSAIC_FILE.value == "mosaic_3d.ome.zarr"
    assert PyxaKeys.MOSAIC_ZIP_FILE.value == "mosaic_3d.ome.zarr.zip"
    assert PyxaKeys.CELL_LABELS.value == "cell_labels"
    assert PyxaKeys.LABEL_ID.value == "label_id"


def test_get_table_counts_are_sparse() -> None:
    from scipy import sparse

    adata = _get_table(FIXTURE_DIR / "cell_by_gene_v1.csv", FIXTURE_DIR / "cell_metadata_v1.csv")
    raw = pd.read_csv(FIXTURE_DIR / "cell_by_gene_v1.csv", index_col="cell_id")
    assert sparse.isspmatrix_csr(adata.X)
    assert adata.X.dtype == raw.to_numpy().dtype
    np.testing.assert_array_equal(adata.X.toarray(), raw.loc[adata.obs_names].to_numpy())
    assert list(adata.var_names) == list(raw.columns)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k "keys_labels or counts_are_sparse" -v`
Expected: FAIL (`AttributeError: MOSAIC_FILE`; `assert False` on `isspmatrix_csr`).

- [ ] **Step 3: Implement**

In `PyxaKeys`, after `UMAP_KEY = "X_umap"`:

```python
    # mosaic image, looked up in the Pyxa directory (unzipped or zipped, as on the Hub)
    MOSAIC_FILE = "mosaic_3d.ome.zarr"
    MOSAIC_ZIP_FILE = "mosaic_3d.ome.zarr.zip"
    # 3D cell labels rasterized from the segmentation polygons onto the mosaic's grid
    CELL_LABELS = "cell_labels"
    LABEL_ID = "label_id"
```

In `pyxa.py`, add `from scipy import sparse` to the imports, and in `_get_table` replace

```python
    adata = ad.AnnData(by_gene, obs=metadata.drop(columns=spatial_cols))
```

with

```python
    # Pyxa counts are mostly zeros: a full Region's dense float64 table is several GB
    adata = ad.AnnData(
        sparse.csr_matrix(by_gene.to_numpy()),
        obs=metadata.drop(columns=spatial_cols),
        var=pd.DataFrame(index=by_gene.columns),
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -q`
Expected: all pass (31 existing + 2 new). If `test_get_table_matches_raw_values` compares `adata.X` densely, change its comparison to `adata.X.toarray()`.

- [ ] **Step 5: Commit**

```bash
git add src/spatialdata_io/_constants/_constants.py src/spatialdata_io/readers/pyxa.py tests/test_pyxa.py
git commit -m "pyxa: sparse counts; keys for the mosaic file and cell labels

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 2: Mosaic lookup and zip, `image=` replaces `image_path`

**Files:**
- Modify: `src/spatialdata_io/readers/pyxa.py` (`_get_image`, new `_open_mosaic`, `_resolve_image`, `pyxa` signature/body/docstring)
- Modify: `src/spatialdata_io/__main__.py` (`pyxa_wrapper`, lines ~913–949)
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `PyxaKeys.MOSAIC_FILE`, `PyxaKeys.MOSAIC_ZIP_FILE` (Task 1).
- Produces: `_open_mosaic(path: Path) -> zarr.Group` (dir or `.zip`, read-only); `_resolve_image(value: InputPath, directory: Path | None) -> Path | None`; `_get_image(path: Path) -> DataTree` accepting a dir or zip; `pyxa(..., *, image: InputPath = None, ...)`.

- [ ] **Step 1: Write the failing tests**

Add a zip helper and tests to `tests/test_pyxa.py`:

```python
import zipfile


def _zip_dir(src: Path, zip_path: Path) -> Path:
    """Zip ``src`` so the archive holds one top-level ``src.name/`` directory, as on the Hub."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent).as_posix())
    return zip_path


def test_get_image_reads_zip_in_place(tmp_path: Path) -> None:
    zipped = _zip_dir(MOSAIC_DIR, tmp_path / "mosaic_3d.ome.zarr.zip")
    from_dir, from_zip = _get_image(MOSAIC_DIR), _get_image(zipped)
    assert list(from_zip.keys()) == list(from_dir.keys())
    for level in from_dir:
        np.testing.assert_array_equal(from_zip[level]["image"].values, from_dir[level]["image"].values)
    assert get_extent(from_zip) == get_extent(from_dir)


def test_pyxa_reader_finds_mosaic(tmp_path: Path) -> None:
    # the fixture holds the unzipped mosaic: found by default, skipped with image=False
    assert "mosaic_image" in pyxa(FIXTURE_DIR, cell_assigned_gene=False).images
    assert not pyxa(FIXTURE_DIR, cell_assigned_gene=False, image=False).images
    # a directory holding only the zip, as downloaded from the Hub
    hub = tmp_path / "hub"
    hub.mkdir()
    for name in ("cell_by_gene_v1.csv", "cell_metadata_v1.csv"):
        (hub / name).write_bytes((FIXTURE_DIR / name).read_bytes())
    _zip_dir(MOSAIC_DIR, hub / "mosaic_3d.ome.zarr.zip")
    assert "mosaic_image" in pyxa(hub).images
    with pytest.raises(FileNotFoundError, match="mosaic image not found"):
        pyxa(hub, image=tmp_path / "nope.ome.zarr")
    empty = tmp_path / "empty"
    empty.mkdir()
    for name in ("cell_by_gene_v1.csv", "cell_metadata_v1.csv"):
        (empty / name).write_bytes((FIXTURE_DIR / name).read_bytes())
    with pytest.raises(FileNotFoundError, match="mosaic_3d.ome.zarr"):
        pyxa(empty, image=True)


def test_pyxa_reader_has_no_image_path() -> None:
    with pytest.raises(TypeError):
        pyxa(FIXTURE_DIR, image_path=MOSAIC_DIR)  # type: ignore[call-arg]
```

Update existing tests that use `image_path` or expect no image:

- `test_pyxa_reader_includes_image_when_given`: `pyxa(FIXTURE_DIR, image=zarr_path)`.
- `test_pyxa_reader_example_mosaic`: `pyxa(FIXTURE_DIR, image=MOSAIC_DIR)`.
- `test_pyxa_reader_missing_image_raises`: `pyxa(FIXTURE_DIR, image=Path(tmpdir) / "does_not_exist.ome.zarr")`.
- `test_pyxa_reader_optional_inputs`: `table_only = pyxa(FIXTURE_DIR, cell_assigned_gene=False, segmentation_geometries=False, pyxa_studio=studio_path, image=False)`; `explicit = pyxa(cell_by_gene=..., cell_metadata=..., image=MOSAIC_DIR)`.
- `test_cli_pyxa`: `"--image", str(f / "mosaic_3d.ome.zarr")` instead of `"--image-path", ...`.
- `test_cli_pyxa_skip`: add `"--no-image"` to the args, and assert `not sdata.images`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -q`
Expected: the new tests and the edited ones FAIL (`TypeError: unexpected keyword argument 'image'`, zip not a directory).

- [ ] **Step 3: Implement in `pyxa.py`**

Add `import zipfile` to the imports. Add above `_get_image`:

```python
def _open_mosaic(path: Path) -> zarr.Group:
    """Open a mosaic OME-Zarr read-only, from its directory or from a zip of it (read in place).

    A zip holds either the group at its root or a single top-level ``<name>.ome.zarr/`` directory,
    as in the Stellaromics/demo dataset.
    """
    if path.suffix != ".zip":
        return zarr.open_group(store=str(path), mode="r")
    with zipfile.ZipFile(path) as zf:
        tops = {name.split("/", 1)[0] for name in zf.namelist()}
    group_path = "" if "zarr.json" in tops or len(tops) != 1 else tops.pop()
    return zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r", path=group_path)
```

In `_get_image`, replace

```python
    group = zarr.open_group(store=str(path), mode="r")
```

with

```python
    group = _open_mosaic(path)
```

and replace

```python
    arrays = [da.squeeze(da.from_zarr(str(path), component=d["path"]), axis=t_index) for d in datasets]
```

with

```python
    arrays = [da.squeeze(da.from_zarr(group[d["path"]]), axis=t_index) for d in datasets]
```

and update its docstring's first line to: `"""Load every level of an OME-Zarr (OME-NGFF v0.5) mosaic, from a directory or a zip, as a multiscale image.`

Add after `_resolve_input`:

```python
def _resolve_image(value: InputPath, path: Path | None) -> Path | None:
    """Where to read the mosaic from, or ``None`` to skip it.

    Like :func:`_resolve_input`, looking in ``path`` for the unzipped ``mosaic_3d.ome.zarr`` first
    and then for ``mosaic_3d.ome.zarr.zip``. An explicit path may be either.
    """
    if value is False:
        return None
    if value is None or value is True:
        for name in (PyxaKeys.MOSAIC_FILE.value, PyxaKeys.MOSAIC_ZIP_FILE.value):
            if path is not None and (path / name).exists():
                return path / name
        if value is True:
            raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {PyxaKeys.MOSAIC_FILE.value}(.zip)")
        return None
    explicit = Path(value)
    if not explicit.exists():
        raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {explicit}")
    return explicit
```

Change the `pyxa` signature to

```python
def pyxa(
    path: str | Path | None = None,
    dataset_id: str = "pyxa",
    *,
    cell_by_gene: RequiredPath = None,
    cell_metadata: RequiredPath = None,
    cell_assigned_gene: InputPath = None,
    segmentation_geometries: InputPath = None,
    pyxa_studio: InputPath = None,
    image: InputPath = None,
) -> SpatialData:
```

In the body, replace

```python
    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {image_path}")
```

with

```python
    image_source = _resolve_image(image, directory)
```

and at the end replace `if image_path is not None: images[...] = _get_image(image_path)` with

```python
    if image_source is not None:
        images[PyxaKeys.MOSAIC_IMAGE.value] = _get_image(image_source)
```

In the docstring, replace the bullet `- A mosaic OME-Zarr image, given as ``image_path``.` with

```
        - ``{px.MOSAIC_FILE!r}`` (or ``{px.MOSAIC_ZIP_FILE!r}``, read in place): the mosaic
          OME-Zarr (OME-NGFF v0.5) image, all pyramid levels, as ``{px.MOSAIC_IMAGE!r}``.
```

remove the `image_path` parameter entry, and add to the optional-files parameter entry (rename it `cell_assigned_gene, segmentation_geometries, pyxa_studio, image`):

```
        ``image`` may be the mosaic's directory or a zip of it; a zipped mosaic is read in place
        with the threaded dask scheduler (a ``ZipStore`` is not shared across processes).
```

- [ ] **Step 4: Implement the CLI in `__main__.py`**

Replace the `--image-path` option and the `image_path` parameter of `pyxa_wrapper`:

```python
@click.option(
    "--image",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    default=None,
    help="Mosaic OME-Zarr directory or zip, if not in the input directory. [default: found in the input]",
)
@click.option("--no-image", is_flag=True, default=False, help="Leave out the mosaic even if present.")
```

and in the function signature `image: str | None = None, no_image: bool = False,` and in the body

```python
    inputs: dict[str, str | bool] = dict.fromkeys(skip, False)
    if pyxa_studio is not None and "pyxa_studio" not in skip:
        inputs["pyxa_studio"] = pyxa_studio
    if no_image:
        inputs["image"] = False
    elif image is not None:
        inputs["image"] = image
    sdata = pyxa(input, dataset_id=dataset_id, **inputs)  # type: ignore[arg-type]
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/spatialdata_io/readers/pyxa.py src/spatialdata_io/__main__.py tests/test_pyxa.py
git commit -m "pyxa: find the mosaic in the directory, zipped or not; image= replaces image_path

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 3: Mosaic grid and label ids

**Files:**
- Create: `src/spatialdata_io/readers/_pyxa_labels.py`
- Modify: `src/spatialdata_io/readers/pyxa.py` (new `_mosaic_grid`; `_get_image` uses it)
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `_open_mosaic(path)` (Task 2).
- Produces:
  - `_MosaicGrid(shapes: tuple[tuple[int, int, int], ...], scale: tuple[float, float, float], translation: tuple[float, float, float])` (frozen dataclass, all z, y, x) with `.transformation -> Sequence` and `.step(level: int) -> tuple[int, int, int]`.
  - `_mosaic_grid(path: Path) -> _MosaicGrid` in `pyxa.py`.
  - `_label_ids(cell_ids: pd.Index) -> tuple[np.ndarray, str]` returning uint32 ids and the rule used.

- [ ] **Step 1: Write the failing tests**

```python
from spatialdata_io.readers._pyxa_labels import _label_ids, _MosaicGrid
from spatialdata_io.readers.pyxa import _mosaic_grid


def test_mosaic_grid_matches_image() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    image = _get_image(MOSAIC_DIR)
    assert grid.shapes == tuple(image[k]["image"].shape[1:] for k in image)
    assert grid.step(0) == (1, 1, 1)
    # (200, 217, 218) -> (12, 14, 13): z 200/12, y 217/14, x 218/13, rounded
    assert grid.step(4) == (17, 16, 17)
    affine = get_transformation(image, to_coordinate_system="global").to_affine_matrix(("z", "y", "x"), ("z", "y", "x"))
    np.testing.assert_allclose(grid.transformation.to_affine_matrix(("z", "y", "x"), ("z", "y", "x")), affine)


def test_label_ids_trailing_integer() -> None:
    ids, rule = _label_ids(pd.Index(["Region_17", "Region_3", "ROI2_40"]))
    assert ids.dtype == np.uint32 and ids.tolist() == [17, 3, 40]
    assert "trailing integer" in rule


@pytest.mark.parametrize(
    ("cell_ids", "why"),
    [
        (["A_1", "B_1"], "not unique"),
        (["Region_1", "Region_x"], "no trailing integer"),
        (["Region_0", "Region_2"], "is 0"),
        (["Region_1", "Region_4294967296"], "2^31"),
    ],
)
def test_label_ids_fallback(cell_ids: list[str], why: str) -> None:
    ids, rule = _label_ids(pd.Index(cell_ids))
    assert ids.tolist() == [1, 2]
    assert why in rule
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k "mosaic_grid or label_ids" -v`
Expected: FAIL with `ModuleNotFoundError: spatialdata_io.readers._pyxa_labels`.

- [ ] **Step 3: Create `_pyxa_labels.py`**

```python
"""3D cell labels for the Pyxa reader, rasterized lazily from the segmentation polygons.

The polygons (one per cell per z-plane, in pixel units) are drawn onto the mosaic image's voxel grid,
so labels and image overlay voxel for voxel at every pyramid level. Ring decoding is eager; drawing is
one ``dask.delayed`` task per tile and runs only when the labels are computed or written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from spatialdata.transformations import Scale, Sequence, Translation

# PIL draws labels into signed 32-bit ("I" mode) images
_MAX_LABEL = 2**31 - 1


@dataclass(frozen=True)
class _MosaicGrid:
    """The mosaic's voxel grid: every level's (z, y, x) shape and level 0's frame in micrometers."""

    shapes: tuple[tuple[int, int, int], ...]
    scale: tuple[float, float, float]
    translation: tuple[float, float, float]

    @property
    def transformation(self) -> Sequence:
        axes = ("z", "y", "x")
        return Sequence([Scale(list(self.scale), axes=axes), Translation(list(self.translation), axes=axes)])

    def step(self, level: int) -> tuple[int, int, int]:
        """Level-0 voxels per voxel of ``level``, per axis (nearest-neighbour stride)."""
        n0, n = self.shapes[0], self.shapes[level]
        return tuple(max(1, round(a / b)) for a, b in zip(n0, n, strict=True))  # type: ignore[return-value]


def _label_ids(cell_ids: pd.Index) -> tuple[np.ndarray, str]:
    """Integer label per cell, and the rule used.

    The trailing integer of each ``cell_id`` (after any prefix, e.g. ``Region_17`` -> 17) when every
    cell has one and they are unique, > 0 (0 is background) and < 2^31; otherwise 1..n in table order.
    """
    n = len(cell_ids)
    digits = pd.Series(np.asarray(cell_ids, dtype=str)).str.extract(r"(\d+)$", expand=False)
    if digits.isna().any():
        why = "no trailing integer in some cell_id"
    elif (digits.str.len() > 10).any() or (numbers := digits.astype(np.int64)).max() > _MAX_LABEL:
        why = "a trailing integer is not below 2^31"
    elif numbers.min() < 1:
        why = "a trailing integer is 0 (the background)"
    elif not numbers.is_unique:
        why = "trailing integers are not unique"
    else:
        return numbers.to_numpy(dtype=np.uint32), "trailing integer of cell_id"
    return np.arange(1, n + 1, dtype=np.uint32), f"1..n in table order ({why})"
```

- [ ] **Step 4: Add `_mosaic_grid` to `pyxa.py` and use it in `_get_image`**

Import at the top of `pyxa.py`: `from spatialdata_io.readers._pyxa_labels import _MosaicGrid`. Add above `_get_image`:

```python
def _mosaic_grid(path: Path) -> _MosaicGrid:
    """The mosaic's level shapes (z, y, x) and level-0 scale and translation, from its OME-NGFF metadata."""
    group = _open_mosaic(path)
    multiscale = cast("dict[str, Any]", group.attrs.asdict()["ome"])["multiscales"][0]
    axes = [a["name"] for a in multiscale["axes"]]
    zyx = [axes.index(a) for a in ("z", "y", "x")]
    datasets = multiscale["datasets"]
    shapes = tuple(tuple(int(group[d["path"]].shape[i]) for i in zyx) for d in datasets)
    transforms = {t["type"]: t for t in datasets[0]["coordinateTransformations"]}
    return _MosaicGrid(
        shapes=shapes,  # type: ignore[arg-type]
        scale=tuple(float(transforms["scale"]["scale"][i]) for i in zyx),  # type: ignore[arg-type]
        translation=tuple(float(transforms["translation"]["translation"][i]) for i in zyx),  # type: ignore[arg-type]
    )
```

In `_get_image`, replace the block from `coordinate_transformations = {...}` through `transformation = Sequence(...)` with

```python
    transformation = _mosaic_grid(path).transformation
    spatial_axes = tuple(a for a in axes if a != "c")
```

(`spatial_axes` is still used for the per-level coords.) Remove `Scale`, `Sequence`, `Translation` from `pyxa.py`'s imports if nothing else uses them.

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -q`
Expected: all pass (including `test_get_image_loads_all_scales`, which checks the tiny store's transform).

- [ ] **Step 6: Commit**

```bash
git add src/spatialdata_io/readers/_pyxa_labels.py src/spatialdata_io/readers/pyxa.py tests/test_pyxa.py
git commit -m "pyxa: mosaic grid shared by image and labels; integer label ids from cell_id

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 4: Decode polygon rings onto the grid

**Files:**
- Modify: `src/spatialdata_io/readers/_pyxa_labels.py`
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `_MosaicGrid` (Task 3); `_get_voxel_size(cell_metadata_path) -> (xy_size, z_size)` (existing, `pyxa.py`).
- Produces:
  - `_Rings` dataclass: `label: np.ndarray[uint32]`, `plane: np.ndarray[int32]` (level-0 plane), `length: np.ndarray[int64]` (vertices per ring), `coords: np.ndarray[float32, (N, 2)]` (x, y in level-0 voxel index space, voxel centres at integers), `bounds: np.ndarray[float32, (n_rings, 4)]` (xmin, ymin, xmax, ymax, same space); `__len__`.
  - `_read_rings(path: Path, labels: pd.Series, grid: _MosaicGrid, xy_size: float, z_size: float, *, simplify: float = 0.25) -> _Rings`, where `labels` maps `cell_id` (index) to uint32 label.

- [ ] **Step 1: Write the failing test**

```python
from spatialdata_io.readers._pyxa_labels import _read_rings


def test_read_rings_on_mosaic_grid() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)

    n_polygons = pq.ParquetFile(FIXTURE_DIR / "segmentation_geometries_v1.parquet").metadata.num_rows
    assert 0 < len(rings) <= 2 * n_polygons  # multipolygons add parts; off-grid planes are dropped
    assert rings.label.dtype == np.uint32 and set(rings.label) <= set(ids)
    nz, ny, nx = grid.shapes[0]
    assert rings.plane.min() >= 0 and rings.plane.max() < nz
    assert rings.length.sum() == len(rings.coords)
    # the fixture's cells lie inside its mosaic crop, up to a cell radius at the edges
    assert rings.bounds[:, 0].min() > -50 and rings.bounds[:, 2].max() < nx + 50
    assert rings.bounds[:, 1].min() > -50 and rings.bounds[:, 3].max() < ny + 50


def test_read_rings_drops_cells_not_in_table() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    one = pd.Series(np.array([7], dtype=np.uint32), index=[cells.iloc[0]])
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", one, grid, xy_size, z_size)
    assert len(rings) > 0 and set(rings.label) == {7}
```

Add `import pyarrow.parquet as pq` to the test imports.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k read_rings -v`
Expected: FAIL with `ImportError: cannot import name '_read_rings'`.

- [ ] **Step 3: Implement**

Add to the imports of `_pyxa_labels.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from spatialdata._logging import logger

from spatialdata_io._constants._constants import PyxaKeys
```

Add:

```python
@dataclass(frozen=True)
class _Rings:
    """Polygon exterior rings in level-0 voxel index space (voxel centres at integer coordinates)."""

    label: np.ndarray
    plane: np.ndarray
    length: np.ndarray
    coords: np.ndarray
    bounds: np.ndarray

    def __len__(self) -> int:
        return len(self.label)


def _rings_from_row_group(
    path: Path,
    row_group: int,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    simplify: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """One parquet row group's rings on the grid, and how many polygon parts were dropped and why."""
    sz, sy, sx = grid.scale
    tz, ty, tx = grid.translation
    nz = grid.shapes[0][0]
    table = pq.ParquetFile(path).read_row_group(
        row_group, columns=[PyxaKeys.CELL_ID.value, PyxaKeys.Z_INDEX.value, "geometry"]
    )
    cell_id = pc.cast(table.column(PyxaKeys.CELL_ID.value), "string").to_numpy(zero_copy_only=False)
    row = labels.index.get_indexer(cell_id)
    zindex = table.column(PyxaKeys.Z_INDEX.value).to_numpy()
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    parts, part_of = shapely.get_parts(geoms, return_index=True)
    rings = shapely.get_exterior_ring(parts)
    rings = shapely.transform(
        rings,
        lambda c: np.column_stack(((c[:, 0] * xy_size - tx) / sx, (c[:, 1] * xy_size - ty) / sy)),
    )
    rings = shapely.simplify(rings, simplify)
    # plane k holds Z_um = (ZIndex + 0.5) * z_size, the reader's plane centre
    plane = np.rint(((zindex[part_of] + 0.5) * z_size - tz) / sz).astype(np.int32)
    in_table = row[part_of] >= 0
    on_grid = (plane >= 0) & (plane < nz)
    empty = shapely.is_empty(rings) | (shapely.get_num_coordinates(rings) < 3)
    keep = in_table & on_grid & ~empty
    rings = rings[keep]
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    out = {
        "label": labels.to_numpy(dtype=np.uint32)[row[part_of][keep]],
        "plane": plane[keep],
        "length": np.bincount(ring_of, minlength=len(rings)).astype(np.int64),
        "coords": coords.astype(np.float32),
        "bounds": shapely.bounds(rings).astype(np.float32).reshape(-1, 4),
    }
    dropped = {
        "not in the table": int((~in_table).sum()),
        "off the mosaic's z range": int((in_table & ~on_grid).sum()),
        "empty": int((in_table & on_grid & empty).sum()),
    }
    return out, dropped


def _read_rings(
    path: Path,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    *,
    simplify: float = 0.25,
) -> _Rings:
    """Every segmentation polygon's exterior ring on the mosaic's level-0 grid, with its label and plane.

    ``labels`` maps ``cell_id`` to the cell's label. Row groups are decoded in a thread pool (pyarrow
    and shapely release the GIL). Rings are simplified to ``simplify`` voxels: the polygons are traced
    on a finer pixel grid than the mosaic's, and the staircase vertices add nothing at its resolution.
    Holes are ignored (exteriors are filled).
    """
    n_groups = pq.ParquetFile(path).metadata.num_row_groups
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda i: _rings_from_row_group(path, i, labels, grid, xy_size, z_size, simplify), range(n_groups)
            )
        )
    parts = [r[0] for r in results]
    dropped: dict[str, int] = {}
    for _, d in results:
        for why, count in d.items():
            dropped[why] = dropped.get(why, 0) + count
    rings = _Rings(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    summary = ", ".join(f"{count} {why}" for why, count in dropped.items() if count)
    logger.info(f"{path.name}: {len(rings)} polygon rings on the mosaic grid" + (f"; dropped {summary}" if summary else ""))
    return rings
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k read_rings -v`
Expected: PASS. If `pc.cast(..., "string")` fails because the column is already `large_string`, use `table.column(...).to_numpy(zero_copy_only=False).astype(str)` instead.

- [ ] **Step 5: Commit**

```bash
git add src/spatialdata_io/readers/_pyxa_labels.py tests/test_pyxa.py
git commit -m "pyxa: decode segmentation polygons into rings on the mosaic grid

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 5: Tiles, tile drawing and one lazy level

**Files:**
- Modify: `src/spatialdata_io/readers/_pyxa_labels.py`
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `_Rings` (Task 4).
- Produces:
  - `_Tile` dataclass: `origin: tuple[int, int, int]` (level voxels), `shape: tuple[int, int, int]`, `step: tuple[int, int]` (dy, dx), `label`, `plane` (tile-local), `offsets` (int64, len n+1), `coords` (float32 (N, 2), level-0 index space).
  - `_plan_tiles(rings: _Rings, shape: tuple[int, int, int], step: tuple[int, int, int], tile: tuple[int, int, int] = TILE) -> list[_Tile]`.
  - `_rasterize_tile(tile: _Tile) -> np.ndarray` (uint32, `tile.shape`).
  - `_labels_level(rings: _Rings, shape: tuple[int, int, int], step: tuple[int, int, int], *, tile: tuple[int, int, int] = TILE, chunks: tuple[int, int, int] = CHUNKS) -> da.Array`.
  - Module constants `TILE = (32, 1024, 1024)`, `CHUNKS = (32, 256, 256)`.

- [ ] **Step 1: Write the failing tests**

```python
from spatialdata_io.readers import _pyxa_labels
from spatialdata_io.readers._pyxa_labels import _labels_level, _plan_tiles, _rasterize_tile, _Rings


def _square_rings(squares: list[tuple[int, int, float, float, float, float]]) -> _Rings:
    """Rings from (label, plane, x0, y0, x1, y1) axis-aligned squares in level-0 voxel index space."""
    coords, bounds = [], []
    for _, _, x0, y0, x1, y1 in squares:
        coords.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float32))
        bounds.append([x0, y0, x1, y1])
    return _Rings(
        label=np.array([s[0] for s in squares], dtype=np.uint32),
        plane=np.array([s[1] for s in squares], dtype=np.int32),
        length=np.full(len(squares), 5, dtype=np.int64),
        coords=np.concatenate(coords),
        bounds=np.array(bounds, dtype=np.float32),
    )


def test_rasterize_square() -> None:
    rings = _square_rings([(5, 1, 2.0, 2.0, 5.0, 5.0)])
    (tile,) = _plan_tiles(rings, (4, 10, 10), (1, 1, 1))
    block = _rasterize_tile(tile)
    assert block.dtype == np.uint32 and block.shape == (4, 10, 10)
    assert (block[1, 2:6, 2:6] == 5).all()  # voxel centres 2..5 lie on or inside the square
    assert block[1, 0, 0] == 0 and block[1, 8, 8] == 0
    assert block[0].max() == 0 and block[2].max() == 0


def test_rasterize_higher_label_wins() -> None:
    for order in ([(3, 0, 0.0, 0.0, 5.0, 5.0), (7, 0, 3.0, 3.0, 8.0, 8.0)],
                  [(7, 0, 3.0, 3.0, 8.0, 8.0), (3, 0, 0.0, 0.0, 5.0, 5.0)]):  # fmt: skip
        (tile,) = _plan_tiles(_square_rings(order), (1, 10, 10), (1, 1, 1))
        block = _rasterize_tile(tile)
        assert block[0, 4, 4] == 7 and block[0, 1, 1] == 3


def test_labels_level_tiles_join_seamlessly() -> None:
    rings = _square_rings([(5, 1, 2.0, 2.0, 7.0, 7.0), (9, 3, 0.0, 6.0, 9.0, 9.0)])
    whole = _rasterize_tile(_plan_tiles(rings, (4, 10, 10), (1, 1, 1))[0])
    tiled = _labels_level(rings, (4, 10, 10), (1, 1, 1), tile=(2, 4, 4), chunks=(2, 3, 3))
    assert tiled.chunksize == (2, 3, 3)
    np.testing.assert_array_equal(tiled.compute(), whole)


def test_labels_level_strides_level_zero() -> None:
    rings = _square_rings([(5, 2, 1.0, 1.0, 12.0, 12.0), (6, 3, 4.0, 4.0, 9.0, 9.0)])
    level0 = _labels_level(rings, (4, 16, 16), (1, 1, 1)).compute()
    level1 = _labels_level(rings, (2, 8, 8), (2, 2, 2)).compute()
    np.testing.assert_array_equal(level1, level0[::2, ::2, ::2])


def test_labels_level_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = _pyxa_labels._rasterize_tile
    monkeypatch.setattr(_pyxa_labels, "_rasterize_tile", lambda tile: calls.append(1) or real(tile))
    array = _labels_level(_square_rings([(5, 0, 1.0, 1.0, 3.0, 3.0)]), (1, 8, 8), (1, 1, 1))
    assert calls == []
    array.compute()
    assert calls == [1]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k "rasterize or labels_level" -v`
Expected: FAIL with `ImportError: cannot import name '_labels_level'`.

- [ ] **Step 3: Implement**

Add to `_pyxa_labels.py` imports: `import dask`, `import dask.array as da`. Add:

```python
# one drawing task: a whole number of storage chunks, so tiles never share a chunk
TILE = (32, 1024, 1024)
# storage chunks, as the mosaic's: an inspect window reads only the chunks it covers
CHUNKS = (32, 256, 256)


@dataclass(frozen=True)
class _Tile:
    """The rings to draw into one tile of one pyramid level."""

    origin: tuple[int, int, int]
    shape: tuple[int, int, int]
    step: tuple[int, int]
    label: np.ndarray
    plane: np.ndarray
    offsets: np.ndarray
    coords: np.ndarray


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Indices of the concatenated slices ``[s, s + n)`` for each (s, n), in order."""
    new_starts = np.cumsum(lengths) - lengths
    return np.arange(int(lengths.sum())) - np.repeat(new_starts, lengths) + np.repeat(starts, lengths)


def _plan_tiles(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    tile: tuple[int, int, int] = TILE,
) -> list[_Tile]:
    """Group the rings of one level (``shape``, ``step`` from level 0) by tile.

    The level keeps the planes a nearest-neighbour stride of level 0 keeps; a ring crossing tiles goes
    to each. Within a tile, rings are ordered by label, so where they overlap the higher label wins.
    """
    nz, ny, nx = shape
    dz, dy, dx = step
    tz, ty, tx = tile
    keep = np.flatnonzero(rings.plane % dz == 0)
    plane = rings.plane[keep] // dz
    on = plane < nz
    keep, plane = keep[on], plane[on]
    # bounds in level-voxel index space; a voxel v is drawn when its centre lies in the ring
    bounds = rings.bounds[keep] / np.array([dx, dy, dx, dy], dtype=np.float32)
    n_ty, n_tx = -(-ny // ty), -(-nx // tx)
    y_lo = np.clip(np.floor(bounds[:, 1] / ty), 0, n_ty - 1).astype(np.int64)
    y_hi = np.clip(np.floor(bounds[:, 3] / ty), 0, n_ty - 1).astype(np.int64)
    x_lo = np.clip(np.floor(bounds[:, 0] / tx), 0, n_tx - 1).astype(np.int64)
    x_hi = np.clip(np.floor(bounds[:, 2] / tx), 0, n_tx - 1).astype(np.int64)
    idx, keys = [], []
    for oy in range(int((y_hi - y_lo).max(initial=0)) + 1):
        for ox in range(int((x_hi - x_lo).max(initial=0)) + 1):
            hit = np.flatnonzero((y_lo + oy <= y_hi) & (x_lo + ox <= x_hi))
            idx.append(hit)
            keys.append(((plane[hit] // tz) * n_ty + y_lo[hit] + oy) * n_tx + x_lo[hit] + ox)
    if not idx:
        return []
    idx, keys = np.concatenate(idx), np.concatenate(keys)
    order = np.lexsort((rings.label[keep][idx], keys))
    idx, keys = idx[order], keys[order]

    ring = keep[idx]
    starts = np.cumsum(rings.length) - rings.length
    lengths = rings.length[ring]
    coords = rings.coords[_ragged_gather(starts[ring], lengths)]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    tiles = []
    edges = np.flatnonzero(np.diff(keys)) + 1
    for lo, hi in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(keys)]]), strict=True):
        kz, rest = divmod(int(keys[lo]), n_ty * n_tx)
        ky, kx = divmod(rest, n_tx)
        origin = (kz * tz, ky * ty, kx * tx)
        c0, c1 = int(offsets[lo]), int(offsets[hi])
        tiles.append(
            _Tile(
                origin=origin,
                shape=(min(tz, nz - origin[0]), min(ty, ny - origin[1]), min(tx, nx - origin[2])),
                step=(dy, dx),
                label=rings.label[ring[lo:hi]],
                plane=plane[idx[lo:hi]] - origin[0],
                offsets=offsets[lo : hi + 1] - c0,
                coords=coords[c0:c1],
            )
        )
    return tiles


def _rasterize_tile(tile: _Tile) -> np.ndarray:
    """Fill the tile's rings plane by plane into a ``uint32`` block (0 = background)."""
    from PIL import Image, ImageDraw

    depth, height, width = tile.shape
    block = np.zeros(tile.shape, dtype=np.uint32)
    dy, dx = tile.step
    _, y0, x0 = tile.origin
    # level voxel index space, then PIL pixel space: pixel p's centre is at p + 0.5
    xy = np.column_stack((tile.coords[:, 0] / dx + 0.5 - x0, tile.coords[:, 1] / dy + 0.5 - y0))
    for plane in np.unique(tile.plane):
        image = Image.new("I", (width, height))
        draw = ImageDraw.Draw(image)
        for r in np.flatnonzero(tile.plane == plane):
            draw.polygon(xy[tile.offsets[r] : tile.offsets[r + 1]].ravel().tolist(), fill=int(tile.label[r]))
        block[plane] = np.asarray(image, dtype=np.int32)
    return block


def _labels_level(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    *,
    tile: tuple[int, int, int] = TILE,
    chunks: tuple[int, int, int] = CHUNKS,
) -> da.Array:
    """One pyramid level as a lazy array: a ``dask.delayed`` drawing task per tile, zeros where no ring falls."""
    by_origin = {t.origin: t for t in _plan_tiles(rings, shape, step, tile)}
    nz, ny, nx = shape
    tz, ty, tx = tile
    blocks = []
    for z in range(0, nz, tz):
        rows = []
        for y in range(0, ny, ty):
            row = []
            for x in range(0, nx, tx):
                size = (min(tz, nz - z), min(ty, ny - y), min(tx, nx - x))
                t = by_origin.get((z, y, x))
                if t is None:
                    row.append(da.zeros(size, dtype=np.uint32, chunks=size))
                else:
                    # looked up at call time, so the drawing function can be patched in tests
                    row.append(da.from_delayed(dask.delayed(_draw)(t), shape=size, dtype=np.uint32))
            rows.append(row)
        blocks.append(rows)
    return da.block(blocks).rechunk(tuple(min(c, s) for c, s in zip(chunks, shape, strict=True)))


def _draw(tile: _Tile) -> np.ndarray:
    return _rasterize_tile(tile)
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k "rasterize or labels_level" -v`
Expected: PASS. If `test_rasterize_square` fails only on the square's edge row/column (PIL's edge rule), keep the interior assertion `block[1, 3:5, 3:5] == 5` and assert the drawn count is between 16 and 25 instead of widening tolerances elsewhere.

- [ ] **Step 5: Commit**

```bash
git add src/spatialdata_io/readers/_pyxa_labels.py tests/test_pyxa.py
git commit -m "pyxa: plan label tiles and draw them lazily, one dask task per tile

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 6: Multiscale labels element

**Files:**
- Modify: `src/spatialdata_io/readers/_pyxa_labels.py`
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `_MosaicGrid`, `_read_rings`, `_labels_level` (Tasks 3–5).
- Produces: `_get_labels(rings: _Rings, grid: _MosaicGrid) -> DataTree`, a validated `Labels3DModel` multiscale with levels `scale0..scaleN` shaped `grid.shapes` and `grid.transformation` on every level.

- [ ] **Step 1: Write the failing tests**

```python
from spatialdata_io.readers._pyxa_labels import _get_labels


def _fixture_labels() -> tuple[DataTree, pd.Series]:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)
    return _get_labels(rings, grid), labels


def test_get_labels_on_mosaic_grid() -> None:
    tree, labels = _fixture_labels()
    image = _get_image(MOSAIC_DIR)
    assert list(tree.keys()) == list(image.keys())
    for level in image:
        assert tree[level]["image"].shape == image[level]["image"].shape[1:]
        assert tree[level]["image"].dtype == np.uint32
    affine = lambda e: get_transformation(e, to_coordinate_system="global").to_affine_matrix(  # noqa: E731
        ("z", "y", "x"), ("z", "y", "x")
    )
    np.testing.assert_allclose(affine(tree), affine(image))
    level0 = tree["scale0"]["image"].values
    assert 0.05 < (level0 > 0).mean() < 0.95
    assert set(np.unique(level0)) - {0} <= set(labels.to_numpy())


def test_get_labels_cell_voxels() -> None:
    """A cell's own polygon centre, on its plane, carries its label."""
    tree, labels = _fixture_labels()
    level0 = tree["scale0"]["image"].values
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    planes = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size, z_size)
    sz, sy, sx = grid.scale
    tz, ty, tx = grid.translation
    checked = 0
    for _, row in planes.sort_values("cell_id").groupby("cell_id").head(1).head(40).iterrows():
        p = row.geometry.representative_point()  # micrometers
        z, y, x = (int(round((v - t) / s)) for v, t, s in ((row["Z_um"], tz, sz), (p.y, ty, sy), (p.x, tx, sx)))
        same_plane = planes[(planes["ZIndex"] == row["ZIndex"]) & (planes["cell_id"] != row["cell_id"])]
        if 0 <= z < level0.shape[0] and not same_plane.geometry.contains(p).any():
            assert level0[z, y, x] == labels[row["cell_id"]]
            checked += 1
    assert checked >= 10


def test_get_labels_levels_stride_level_zero() -> None:
    tree, _ = _fixture_labels()
    grid = _mosaic_grid(MOSAIC_DIR)
    level0 = tree["scale0"]["image"].values
    for i in range(1, len(grid.shapes)):
        dz, dy, dx = grid.step(i)
        nz, ny, nx = grid.shapes[i]
        strided = level0[::dz, ::dy, ::dx][:nz, :ny, :nx]
        level = tree[f"scale{i}"]["image"].values
        # identical except where PIL's edge rule differs between scales
        assert (level == strided).mean() > 0.97
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k get_labels -v`
Expected: FAIL with `ImportError: cannot import name '_get_labels'`.

- [ ] **Step 3: Implement**

Add to `_pyxa_labels.py` imports: `from spatialdata.models import Labels3DModel`, `from spatialdata.transformations import set_transformation`, `from xarray import DataArray, Dataset, DataTree`. Add:

```python
def _get_labels(rings: _Rings, grid: _MosaicGrid) -> DataTree:
    """The cells as a lazy multiscale ``Labels3DModel`` on the mosaic's grid, one level per mosaic level."""
    n0 = grid.shapes[0]
    levels = {}
    for i, shape in enumerate(grid.shapes):
        array = _labels_level(rings, shape, grid.step(i))
        # coordinates of every level are pixel centres in scale0 pixel units, as spatialdata assigns them
        coords = {ax: np.linspace(0, a, b + 1)[:-1] + a / b / 2 for ax, a, b in zip("zyx", n0, shape, strict=True)}
        levels[f"scale{i}"] = Dataset({"image": DataArray(array, dims=("z", "y", "x"), coords=coords)})
    tree = DataTree.from_dict(levels)
    set_transformation(tree, {"global": grid.transformation}, set_all=True)
    Labels3DModel.validate(tree)
    logger.info(f"{PyxaKeys.CELL_LABELS.value}: {len(grid.shapes)} levels planned; drawn when computed or written")
    return tree
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k get_labels -v`
Expected: PASS. If `test_get_labels_cell_voxels` fails for a few cells whose representative point sits within half a voxel of the boundary, skip points whose `row.geometry.boundary.distance(p) < 0.5 * sx` rather than loosening the equality.

- [ ] **Step 5: Commit**

```bash
git add src/spatialdata_io/readers/_pyxa_labels.py tests/test_pyxa.py
git commit -m "pyxa: multiscale 3D cell labels on the mosaic grid

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 7: Wire `labels=` and `shapes=` into `pyxa()` and the CLI

**Files:**
- Modify: `src/spatialdata_io/readers/pyxa.py` (`pyxa` signature, body, docstring)
- Modify: `src/spatialdata_io/__main__.py` (`pyxa_wrapper`)
- Modify: `docs/changelog.md`, `README.md` (Pyxa entry)
- Test: `tests/test_pyxa.py`

**Interfaces:**
- Consumes: `_resolve_image`, `_mosaic_grid`, `_get_image` (Tasks 2–3); `_label_ids`, `_read_rings`, `_get_labels` (Tasks 3–6); `PyxaKeys.CELL_LABELS`, `PyxaKeys.LABEL_ID`.
- Produces: `pyxa(..., *, image: InputPath = None, shapes: bool | None = None, labels: bool = False) -> SpatialData`; CLI flags `--labels`, `--shapes/--no-shapes`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pyxa_reader_labels(tmp_path: Path) -> None:
    sdata = pyxa(FIXTURE_DIR, cell_assigned_gene=False, labels=True)
    assert set(sdata.labels) == {"cell_labels"} and set(sdata.images) == {"mosaic_image"}
    assert not sdata.shapes  # labels replace the shapes by default
    table = sdata["rna"]
    assert get_table_keys(table) == ("cell_labels", "region", "label_id")
    assert table.obs["label_id"].dtype == np.uint32
    assert "cell_id" in table.obs

    sdata.write(tmp_path / "labels.zarr")
    back = read_zarr(tmp_path / "labels.zarr")
    drawn = set(np.unique(back["cell_labels"]["scale0"]["image"].values)) - {0}
    assert drawn <= set(back["rna"].obs["label_id"])
    assert get_table_keys(back["rna"])[0] == "cell_labels"


def test_pyxa_reader_labels_and_shapes() -> None:
    sdata = pyxa(FIXTURE_DIR, cell_assigned_gene=False, labels=True, shapes=True)
    assert set(sdata.shapes) == {"cell_boundaries", "cell_boundaries_z"}
    assert get_table_keys(sdata["rna"])[0] == "cell_labels"
    assert not pyxa(FIXTURE_DIR, cell_assigned_gene=False, shapes=False).shapes


def test_pyxa_reader_labels_need_image_and_geometries() -> None:
    with pytest.raises(ValueError, match="missing: a mosaic image"):
        pyxa(FIXTURE_DIR, labels=True, image=False)
    with pytest.raises(ValueError, match="missing: segmentation_geometries"):
        pyxa(FIXTURE_DIR, labels=True, segmentation_geometries=False)
    with pytest.raises(FileNotFoundError, match="segmentation_geometries_v1.parquet"):
        pyxa(FIXTURE_DIR, shapes=True, segmentation_geometries=False)


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa_labels(dataset: str, tmp_path: Path) -> None:
    output_zarr = tmp_path / "data.zarr"
    result = CliRunner().invoke(
        pyxa_wrapper,
        ["--input", str(Path("./data") / dataset), "--output", str(output_zarr),
         "--skip", "cell_assigned_gene", "--labels"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    sdata = read_zarr(output_zarr)
    assert set(sdata.labels) == {"cell_labels"} and not sdata.shapes
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -k "labels" -v`
Expected: FAIL with `TypeError: pyxa() got an unexpected keyword argument 'labels'`.

- [ ] **Step 3: Implement in `pyxa.py`**

Import: `from spatialdata.models import Image3DModel, PointsModel, ShapesModel, TableModel` stays; add `from spatialdata_io.readers._pyxa_labels import _get_labels, _label_ids, _MosaicGrid, _read_rings`.

Signature: add after `image: InputPath = None,`

```python
    shapes: bool | None = None,
    labels: bool = False,
```

Replace the body from `points = {}` through the final `return` with:

```python
    if labels:
        missing = [
            name
            for name, found in (("segmentation_geometries", geometries_path), ("a mosaic image", image_source))
            if found is None
        ]
        if missing:
            raise ValueError(
                f"labels=True needs segmentation_geometries and a mosaic image; missing: {', '.join(missing)}"
            )
    if shapes and geometries_path is None:
        raise FileNotFoundError(
            f"Expected Pyxa output file not found: {PyxaKeys.SEGMENTATION_GEOMETRIES_FILE.value} (shapes=True)"
        )
    read_shapes = geometries_path is not None and (shapes if shapes is not None else not labels)

    points = {}
    if assigned_gene_path is not None:
        points["transcripts"] = PointsModel.parse(
            _get_points(assigned_gene_path),
            coordinates={"x": PyxaKeys.X_UM.value, "y": PyxaKeys.Y_UM.value, "z": PyxaKeys.Z_UM.value},
            feature_key=PyxaKeys.GENE.value,
            instance_key=PyxaKeys.CELL_ID.value,
        )

    xy_size, z_size = _get_voxel_size(metadata_path) if (read_shapes or labels) else (1.0, 1.0)
    shapes_elements = {}
    if read_shapes:
        planes = _get_shapes(geometries_path, xy_size, z_size)  # type: ignore[arg-type]
        shapes_elements[PyxaKeys.REGION.value] = ShapesModel.parse(_get_footprints(planes))
        shapes_elements[PyxaKeys.CELL_BOUNDARIES_Z.value] = ShapesModel.parse(planes)

    adata = _get_table(by_gene_path, metadata_path, studio_path)
    labels_elements = {}
    if labels:
        ids, rule = _label_ids(adata.obs_names)
        logger.info(f"{PyxaKeys.LABEL_ID.value}: {rule}")
        adata.obs[PyxaKeys.LABEL_ID.value] = ids
        adata.obs[PyxaKeys.REGION_KEY.value] = pd.Series(
            PyxaKeys.CELL_LABELS.value, index=adata.obs_names, dtype="category"
        )
        grid = _mosaic_grid(image_source)  # type: ignore[arg-type]
        rings = _read_rings(
            geometries_path,  # type: ignore[arg-type]
            pd.Series(ids, index=adata.obs_names),
            grid,
            xy_size,
            z_size,
        )
        labels_elements[PyxaKeys.CELL_LABELS.value] = _get_labels(rings, grid)
        table = TableModel.parse(
            adata,
            region=PyxaKeys.CELL_LABELS.value,
            region_key=PyxaKeys.REGION_KEY.value,
            instance_key=PyxaKeys.LABEL_ID.value,
        )
    elif shapes_elements:
        table = TableModel.parse(
            adata,
            region=PyxaKeys.REGION.value,
            region_key=PyxaKeys.REGION_KEY.value,
            instance_key=PyxaKeys.INSTANCE_KEY.value,
        )
    else:
        table = TableModel.parse(adata)

    images = {}
    if image_source is not None:
        images[PyxaKeys.MOSAIC_IMAGE.value] = _get_image(image_source)

    return SpatialData(
        points=points, shapes=shapes_elements, labels=labels_elements, tables={"rna": table}, images=images
    )
```

Docstring: after the "The polygons are returned as two shapes elements" block, add

```
    With ``labels=True`` the polygons are instead rasterized into ``{px.CELL_LABELS!r}``, a 3D
    labels element on the mosaic's voxel grid (same pyramid levels and transformation as
    ``{px.MOSAIC_IMAGE!r}``), and the table annotates it through the integer
    ``{px.LABEL_ID!r}``: the trailing integer of each ``cell_id`` (``Region_17`` -> 17) when
    those are unique, positive and below 2^31, otherwise 1..n in table order. Holes are
    filled, and where two cells overlap on a plane the higher label wins. The labels are
    lazy: they are drawn, one task per 32 x 1024 x 1024 tile, when computed or written, with
    dask's default threaded scheduler (a process scheduler is much slower here, since every
    drawn tile is pickled back). Decoding the polygons is eager: for a full Region (23M
    polygons) about a minute and ~6 GB of memory.
```

and the parameters:

```
    shapes
        Return the polygons as shapes. ``None`` (default): when the segmentation geometries are
        read and ``labels`` is ``False``.
    labels
        Rasterize the polygons into 3D cell labels on the mosaic's grid (needs the segmentation
        geometries and the mosaic); the table then annotates the labels.
```

- [ ] **Step 4: CLI flags in `__main__.py`**

Add options to `pyxa_wrapper`:

```python
@click.option("--labels", is_flag=True, default=False, help="Rasterize 3D cell labels onto the mosaic's grid.")
@click.option(
    "--shapes/--no-shapes",
    default=None,
    help="Return the polygons as shapes. [default: when read and --labels is not set]",
)
```

signature `labels: bool = False, shapes: bool | None = None,` and pass `labels=labels, shapes=shapes` to `pyxa(...)`.

- [ ] **Step 5: Docs**

In `docs/changelog.md`, under the unreleased section's Pyxa entry (add one if absent), add:

```
- `pyxa`: `labels=True` rasterizes the segmentation polygons into lazy 3D `cell_labels` on the mosaic's grid, annotated by the table; `shapes=` controls the shapes; the mosaic is found in the Pyxa directory (`mosaic_3d.ome.zarr`, or its `.zip` read in place) via `image=`, which replaces `image_path`; counts are sparse.
```

In `README.md`'s Pyxa line (line ~57), append: `` `labels=True` adds 3D cell labels on the mosaic grid.``

- [ ] **Step 6: Run the full suite and lint**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pyxa.py -q` then `.venv/Scripts/python.exe -m ruff check src/spatialdata_io/readers/pyxa.py src/spatialdata_io/readers/_pyxa_labels.py src/spatialdata_io/__main__.py tests/test_pyxa.py` then `.venv/Scripts/python.exe -m mypy src/spatialdata_io/readers/pyxa.py src/spatialdata_io/readers/_pyxa_labels.py`
Expected: tests pass; ruff clean; mypy no new errors (fix any by typing, not by blanket ignores).

- [ ] **Step 7: Commit**

```bash
git add src/spatialdata_io/readers/pyxa.py src/spatialdata_io/__main__.py tests/test_pyxa.py docs/changelog.md README.md
git commit -m "pyxa: labels= rasterizes 3D cell labels annotated by the table; shapes= option

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

---

### Task 8: Full-region check on colon A2 (manual, not CI)

**Files:**
- Create (scratchpad only, not committed): `C:\Users\BOXX\AppData\Local\Temp\claude\D--clarence-pyxa-scverse-demo\fc9c2339-d9c9-430f-8723-115ce48a4a23\scratchpad\colon_reader_check.py`

**Interfaces:**
- Consumes: `pyxa(..., labels=True, image=...)` (Task 7).
- Produces: timings and a voxel comparison to report in the PR.

- [ ] **Step 1: Write the check script**

```python
import time
from pathlib import Path

import numpy as np
import zarr
from spatialdata import read_zarr
from spatialdata_io.experimental import pyxa

SRC = Path(r"D:/clarence/20260818_glasgow_colon_h1k/Run01/Pyxa_results/Analysis02/A2/Analysis Group")
OUT = Path(r"D:/clarence/colon_reader_check.sdata.zarr")
OLD = Path(r"D:/clarence/pyxa_scverse_demo/data/colon_a2.sdata.zarr")

t0 = time.perf_counter()
sdata = pyxa(SRC / "ag_output", labels=True, cell_assigned_gene=False, image=SRC / "Region/mosaic/mosaic_3d.ome.zarr")
t1 = time.perf_counter()
sdata.write(OUT, overwrite=True)
t2 = time.perf_counter()
print(f"read {t1 - t0:.0f} s, write {t2 - t1:.0f} s")
new = read_zarr(OUT)["cell_labels"]["scale0"]["image"]
old = read_zarr(OLD)["cell_labels"]["scale0"]["image"]
window = (slice(100, 132), slice(4096, 5120), slice(5120, 6144))
a, b = new.data[window].compute(), old.data[window].compute()
print("level-0 window agreement with the demo build:", float((a == b).mean()))
```

- [ ] **Step 2: Run it**

Run: `D:\clarence\spatialdata-io\.venv\Scripts\python.exe <scratchpad>\colon_reader_check.py` (takes several minutes; run in the background).
Expected: read ≈ 1.5 min (ring decoding + table), write ≈ 5–10 min (labels ≈ 3 min + mosaic copy), window agreement ≥ 0.999 (the demo build drew rings in label order within tiles too).

- [ ] **Step 3: Record and clean up**

Note the timings and agreement for the PR description; delete `D:/clarence/colon_reader_check.sdata.zarr`.

---

### Task 9: Demo uses the reader

**Files:**
- Modify: `D:\clarence\pyxa_scverse_demo\build_colon_a2.py` (rewrite)
- Modify: `D:\clarence\pyxa_scverse_demo\demo_pyxa.ipynb` (`image_path=` → `image=`)
- Modify: `D:\clarence\pyxa_scverse_demo\README.md` (build section)

**Interfaces:**
- Consumes: `pyxa(..., labels=True)` (Task 7), element names `mosaic_image`, `cell_labels`, table `rna`.

- [ ] **Step 1: Rewrite `build_colon_a2.py`**

```python
"""Build the Glasgow colon A2 SpatialData for ``colon_a2.py`` with spatialdata-io's ``pyxa`` reader.

``--download`` fetches the region from the ``colon/`` folder of the
`Stellaromics/demo <https://huggingface.co/datasets/Stellaromics/demo>`_ dataset into
``data/hf/colon/`` (about 22 GB; the transcripts stay on the Hub). ``--source`` also takes a
Pyxa ``Analysis Group`` directory. The reader returns the table, the mosaic (read from its zip
in place) and 3D cell labels drawn onto the mosaic's grid; this script keeps the cells Pyxa
Studio kept and writes ``data/colon_a2.sdata.zarr``.

    uv run python build_colon_a2.py --download --overwrite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spatialdata_io.experimental import pyxa

DATA = Path(__file__).resolve().parent / "data"
HF_REPO = "Stellaromics/demo"
HF_FOLDER = "colon"
DEFAULT_SOURCE = DATA / "hf" / HF_FOLDER
OUT = DATA / "colon_a2.sdata.zarr"
# What the build reads from the Hub; the transcripts (cell_assigned_gene_v1.csv) stay there.
HF_FILES = [
    "cell_by_gene_v1.csv",
    "cell_metadata_v1.csv",
    "pyxa_studio_v1.csv",
    "segmentation_geometries_v1.parquet",
    "mosaic_3d.ome.zarr.zip",
]


def download(dest: Path = DEFAULT_SOURCE) -> Path:
    """Fetch the colon region from the Hub into ``dest``."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_REPO, repo_type="dataset", allow_patterns=[f"{HF_FOLDER}/{f}" for f in HF_FILES], local_dir=dest.parent
    )
    return dest.parent / HF_FOLDER


def read(source: Path):
    """The region as a SpatialData; an ``Analysis Group`` keeps its tables and mosaic apart."""
    if (source / "ag_output").is_dir():
        return pyxa(
            source / "ag_output",
            cell_assigned_gene=False,
            labels=True,
            image=source / "Region" / "mosaic" / "mosaic_3d.ome.zarr",
        )
    return pyxa(source, cell_assigned_gene=False, labels=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Hub folder or Pyxa 'Analysis Group'")
    parser.add_argument("--download", action="store_true", help=f"fetch {HF_REPO}/{HF_FOLDER} into --source first")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.download:
        args.source = download(args.source)

    sdata = read(args.source)
    table = sdata.tables["rna"]
    # Studio drops cells that fail Pyxa's filters; they carry no cluster. Their labels stay drawn.
    sdata.tables["rna"] = table[table.obs["Cluster"].notna()].copy()
    t0 = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(args.out, overwrite=args.overwrite)
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f} s", flush=True)
    kept = sdata.tables["rna"]
    print(json.dumps({"out": str(args.out), "n_obs": kept.n_obs, "labels": list(sdata.labels), "images": list(sdata.images)}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update `demo_pyxa.ipynb` and README**

In `demo_pyxa.ipynb`, replace `image_path=` with `image=` in the `pyxa(...)` call (and drop the unzip cell if it only extracted the mosaic for `image_path`). In `README.md`'s colon build section: the build is now one `pyxa(..., labels=True)` call; drop the unzip step, the `--no-labels` / `--no-mosaic` / `--only-mosaic` flags and the rasterization description; keep `--download`, `--source`, `--out`, `--overwrite`; sizes: ~22 GB download, ~13 GB output, build ≈ 10 min.

- [ ] **Step 3: Build and run the notebook**

Run: `cd D:\clarence\pyxa_scverse_demo; uv run python build_colon_a2.py --source "D:/clarence/20260818_glasgow_colon_h1k/Run01/Pyxa_results/Analysis02/A2/Analysis Group" --overwrite` (background), then the headless check from the earlier session (`from colon_a2 import app; app.run()` with `MPLBACKEND=Agg`).
Expected: `n_obs` 358173, labels `["cell_labels"]`, images `["mosaic_image"]`; notebook runs; `widget.volume` has both `image_url` and `labels_url`.

- [ ] **Step 4: Commit**

```bash
git add build_colon_a2.py demo_pyxa.ipynb README.md
git commit -m "Colon A2 build is one pyxa(labels=True) call

Co-authored-by: ckmah <clarence.mah@stellaromics.com>"
```

- [ ] **Step 5: Ask before publishing**

Ask the user before pushing either repo and before updating the Stellaromics/demo dataset card's usage snippet (`image=` in place of the unzip step; `labels=True` for `colon/`).
