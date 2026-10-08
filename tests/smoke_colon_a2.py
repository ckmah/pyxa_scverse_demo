"""Headless smoke test for ``colon_a2.py`` on a small synthetic SpatialData.

The real store is ~13 GB and needs ~30 GB of memory to build, so this writes a stand-in
table from the committed annotations instead: a random subset of cells at their real
µm positions, with Poisson counts driven by each cell's Novae niche signature genes. It
then runs every notebook cell twice and fails on any error: once with only the demo
landmarks (vignette 2 takes its Python fallback), once with a promoted neighborhood
selection on the widget (vignette 2 reads the widget selection).

    uv run python tests/smoke_colon_a2.py            # run all cells
    uv run python tests/smoke_colon_a2.py --html out.html   # also export HTML
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations" / "colon_a2"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", type=Path, help="also export the rendered notebook here")
    parser.add_argument("--n-cells", type=int, default=40_000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "colon_a2_smoke.sdata.zarr"
        make_sdata(store, args.n_cells)
        os.environ["COLON_A2_SDATA"] = str(store)
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        from colon_a2 import app

        _, defs = app.run()
        print(f"ran colon_a2.py: {defs['adata'].n_obs:,} cells, {len(defs['landmarks'].landmarks)} landmarks")

        # Second pass: a widget selection standing in for a promoted neighborhood.
        import milume

        init = milume.LandmarksWidget.__init__

        def with_selection(self, *a, **kw):
            init(self, *a, **kw)
            picked = np.random.default_rng(1).choice(self._data_x.shape[0], 2_000, replace=False)
            self.selections = [{"id": "neighborhood-1", "type": "lasso", "point_indices": picked.tolist()}]

        milume.LandmarksWidget.__init__ = with_selection
        try:
            _, defs = app.run()
        finally:
            milume.LandmarksWidget.__init__ = init
        assert defs["sel2_pick"].value == "neighborhood-1", defs["sel2_pick"].value
        print("ran colon_a2.py with a widget neighborhood selection")
        if args.html:
            subprocess.run(
                [sys.executable, "-m", "marimo", "export", "html", "colon_a2.py", "-o", str(args.html), "--force", "--no-include-code"],
                check=True,
                env=os.environ,
            )
            text = Path(args.html).read_text()
            if "marimo-error" in text or "Traceback" in text:
                sys.exit(f"export of colon_a2.py rendered an error; see {args.html}")
            print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
