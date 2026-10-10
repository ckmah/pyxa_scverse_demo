"""Headless smoke test for ``3_colon_3d_milume.py`` on a small synthetic SpatialData.

The real store is ~13 GB and needs ~30 GB of memory to build, so this writes a stand-in
table from the committed annotations instead: a random subset of cells at their real
µm positions, with Poisson counts driven by each cell's Novae niche signature genes. It
then runs every notebook cell twice and fails on any error: once with no landmarks (each
vignette shows its drawing instructions), once with a shape and a buffered line injected
as if drawn in the widget (each vignette plots).

    uv run python tests/smoke_3_colon_3d_milume.py                    # run all cells
    uv run python tests/smoke_3_colon_3d_milume.py --html out.html    # also export HTML (with landmarks)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations" / "colon_a2"

# Landmarks as a user would draw them on colon A2 (µm): a square over clean normal crypt and a
# line from tumour into stroma with a 150 µm buffer.
USER_LANDMARKS = [
    {
        "id": "user-shape",
        "type": "shape",
        "vertices": [[917.0, 2796.0], [1217.0, 2796.0], [1217.0, 3096.0], [917.0, 3096.0]],
    },
    {
        "id": "user-line",
        "type": "line",
        "vertices": [[-550.0, 2850.0], [-1138.0, 2732.0]],
        "buffer_width": 150.0,
        "buffer_side": "both",
    },
]


def make_sdata(path: Path, n_cells: int = 40_000, seed: int = 0) -> None:
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    rng = np.random.default_rng(seed)
    typing = pd.read_parquet(ANN / "cell_typing.parquet", columns=["cell_id", "X_um", "Y_um", "Z_um"])
    domains = pd.read_parquet(ANN / "novae_domains.parquet", columns=["cell_id", "domain_L10"])
    niches = pd.read_csv(ANN / "niche_signatures.csv").set_index("domain")
    cells = typing.merge(domains, on="cell_id").sample(n_cells, random_state=seed).reset_index(drop=True)
    signature = {
        d: [g.split(" (")[0] for g in str(top).split("; ")] for d, top in niches["top_genes"].items()
    }
    genes = list(dict.fromkeys(g for gs in signature.values() for g in gs))
    genes += [f"Filler{i:02d}" for i in range(20)]
    col = {g: i for i, g in enumerate(genes)}
    rate = np.full((n_cells, len(genes)), 0.3)
    for d, gs in signature.items():
        rows = np.flatnonzero(cells["domain_L10"].to_numpy() == d)
        rate[np.ix_(rows, [col[g] for g in gs])] += rng.uniform(2, 6, len(gs))
    counts = sp.csr_matrix(rng.poisson(rate).astype(np.int64))
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()}, index=cells["cell_id"].to_numpy()),
        var=pd.DataFrame(index=genes),
        obsm={"spatial": cells[["X_um", "Y_um", "Z_um"]].to_numpy(float)},
    )
    SpatialData(tables={"rna": TableModel.parse(adata)}).write(path, overwrite=True)


@contextmanager
def drawn(landmarks: list[dict]):
    """Make ``milume.peek`` return a widget that already holds ``landmarks``, as if drawn."""
    import milume

    peek = milume.peek

    def with_landmarks(*args, **kwargs):
        widget = peek(*args, **kwargs)
        widget.landmarks = [dict(lm) for lm in landmarks]
        return widget

    milume.peek = with_landmarks
    try:
        yield
    finally:
        milume.peek = peek


def run(app, landmarks: list[dict]) -> dict:
    with drawn(landmarks):
        _, defs = app.run()
    return defs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", type=Path, help="also export the rendered notebook (with landmarks) here")
    parser.add_argument("--n-cells", type=int, default=40_000)
    parser.add_argument("--sdata", type=Path, help="run on this store instead of synthetic data")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        store = args.sdata or Path(tmp) / "colon_a2_smoke.sdata.zarr"
        if args.sdata is None:
            make_sdata(store, args.n_cells)
        os.environ["COLON_A2_SDATA"] = str(store)
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        from importlib import import_module
        app = import_module("3_colon_3d_milume").app

        defs = run(app, [])
        assert defs["shape_pick"].value == "(none)" and defs["line_pick"].value == "(none)"
        print(f"no landmarks: ran 3_colon_3d_milume.py on {defs['adata'].n_obs:,} cells; both vignettes show instructions")

        unbuffered = [dict(USER_LANDMARKS[1], buffer_width=0.0)]
        defs = run(app, unbuffered)
        assert defs["line_pick"].value == "(none)"
        print("unbuffered line: vignette 2 asks for a buffer")

        defs = run(app, USER_LANDMARKS)
        assert defs["shape_pick"].value == "user-shape" and defs["line_pick"].value == "user-line"
        print("drawn shape + buffered line: both vignettes plot")

        if args.html:
            # marimo export runs the notebook in its own kernel, so export a temporary copy whose
            # widget cell adds the landmarks instead of patching the class.
            source = (ROOT / "3_colon_3d_milume.py").read_text()
            hook = "    landmarks = mo.ui.anywidget(widget)\n"
            assert hook in source
            copy = ROOT / "_colon_a2_export.py"
            copy.write_text(source.replace(hook, f"    widget.landmarks = {USER_LANDMARKS!r}\n" + hook))
            try:
                subprocess.run(
                    [sys.executable, "-m", "marimo", "export", "html", str(copy), "-o", str(args.html), "--force",
                     "--no-include-code"],
                    check=True,
                    env=os.environ,
                )
            finally:
                copy.unlink()
            text = Path(args.html).read_text()
            if "marimo-error" in text or "Traceback" in text:
                sys.exit(f"export of 3_colon_3d_milume.py rendered an error; see {args.html}")
            print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
