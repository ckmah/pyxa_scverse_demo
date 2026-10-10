"""Build the Glasgow colon A2 SpatialData for ``3_colon_3d_milume.py`` with spatialdata-io's ``pyxa`` reader.

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

from spatialdata import SpatialData
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


def read(source: Path) -> SpatialData:
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
    summary = {"out": str(args.out), "n_obs": kept.n_obs, "labels": list(sdata.labels), "images": list(sdata.images)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
