"""
Export real signals from the notebook's dataset into sample_signals/.

The notebook reads X_test.pkl / y_test.pkl from Google Drive. If you have those
files locally, this pulls a handful of individual recordings out so the app ships
with real demo data instead of the synthetic stand-ins.

    python export_samples.py --x /path/to/X_test.pkl --y /path/to/y_test.pkl -n 6

Label convention, straight from the notebook: the raw pickle uses 0 = ALS,
1 = Normal, and the notebook flips it (`y_fixed = 1 - y`) so that 1 = ALS.
The exported filenames use the flipped, human convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", required=True, type=Path, help="X_test.pkl")
    ap.add_argument("--y", type=Path, default=None, help="y_test.pkl (optional, for labels)")
    ap.add_argument("-n", "--count", type=int, default=6, help="signals to export")
    ap.add_argument("--out", type=Path, default=config.SAMPLE_DIR)
    args = ap.parse_args()

    X = np.array(pd.read_pickle(args.x))
    if X.ndim == 3:
        X = X.reshape(X.shape[0], X.shape[1])

    y = None
    if args.y is not None:
        y_raw = np.array(pd.read_pickle(args.y)).reshape(-1)
        y = 1 - y_raw  # notebook convention: 1 = ALS after the flip

    args.out.mkdir(parents=True, exist_ok=True)

    if y is not None:
        als_idx = np.flatnonzero(y == 1)[: args.count // 2]
        normal_idx = np.flatnonzero(y == 0)[: args.count - len(als_idx)]
        chosen = [(int(i), "als") for i in als_idx] + [(int(i), "normal") for i in normal_idx]
    else:
        chosen = [(i, "unlabeled") for i in range(min(args.count, len(X)))]

    for idx, label in chosen:
        path = args.out / f"real_{label}_{idx + 1:04d}.npy"
        np.save(path, X[idx].astype(np.float32).ravel())
        print(f"  {path.name}  ({X[idx].size:,} points)")

    print(f"\nExported {len(chosen)} signals to {args.out}")
    print("The label in the filename is ground truth, not a prediction.")


if __name__ == "__main__":
    main()
