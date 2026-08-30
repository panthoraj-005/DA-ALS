"""
Generate bundled demo signals so the app is testable end to end without the
original dataset.

These are SYNTHETIC. They are shaped like the real inputs (23,437 points, the
same amplitude range) and are enough to exercise upload -> predict -> report,
but a prediction made on them means nothing clinically. Filenames say so.

Real signals: if you have the notebook's X_test.pkl, use export_samples.py
instead - those are the recordings the models were actually evaluated on.

    python make_samples.py
"""

from __future__ import annotations

import numpy as np

import config

RNG_SEED = 7


def _dense_interference(n: int, rng: np.random.Generator) -> np.ndarray:
    """Normal-like: dense, low-amplitude, many small overlapping units."""
    sig = rng.normal(0, 1.0, n)
    kernel = np.hanning(9)
    sig = np.convolve(sig, kernel / kernel.sum(), mode="same")
    for _ in range(1400):
        at = rng.integers(0, n - 40)
        width = int(rng.integers(8, 22))
        amp = rng.normal(0, 1.4)
        sig[at : at + width] += amp * np.hanning(width)
    return sig


def _reduced_recruitment(n: int, rng: np.random.Generator) -> np.ndarray:
    """ALS-like: sparse, large, long-duration units on a quieter baseline."""
    sig = rng.normal(0, 0.45, n)
    for _ in range(180):
        at = rng.integers(0, n - 200)
        width = int(rng.integers(60, 190))
        amp = rng.normal(0, 6.5)
        shape = np.hanning(width) * np.sin(np.linspace(0, np.pi * 3, width))
        sig[at : at + width] += amp * shape
    for _ in range(30):  # bursts of fasciculation-like activity
        at = rng.integers(0, n - 900)
        width = int(rng.integers(300, 900))
        sig[at : at + width] *= rng.uniform(1.8, 3.2)
    return sig


def main() -> None:
    n = config.SIGNAL_LENGTH
    rng = np.random.default_rng(RNG_SEED)
    out = config.SAMPLE_DIR
    out.mkdir(parents=True, exist_ok=True)

    built = []
    for i in (1, 2):
        sig = _reduced_recruitment(n, rng).astype(np.float32)
        path = out / f"synthetic_als_like_{i}.npy"
        np.save(path, sig)
        built.append(path)

    for i in (1, 2):
        sig = _dense_interference(n, rng).astype(np.float32)
        path = out / f"synthetic_normal_like_{i}.npy"
        np.save(path, sig)
        built.append(path)

    # One CSV so the CSV reader has something to exercise too.
    csv_path = out / "synthetic_normal_like_1.csv"
    np.savetxt(csv_path, np.load(out / "synthetic_normal_like_1.npy"), delimiter=",", fmt="%.6f")
    built.append(csv_path)

    print(f"Wrote {len(built)} demo signals of {n:,} points to {out}")
    for p in built:
        print(f"  {p.name}")
    print("\nThese are synthetic. Predictions on them are not clinically meaningful.")


if __name__ == "__main__":
    main()
