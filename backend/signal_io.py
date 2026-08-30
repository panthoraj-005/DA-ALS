"""
Reading uploaded signals (.npy / .csv / .pkl) and validating their shape.

Validation failures raise SignalFormatError, which main.py turns into a 400 with
a message that says what was expected — never a stack trace.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

import config

SUPPORTED_SUFFIXES = {".npy", ".csv", ".txt", ".pkl", ".pickle"}


class SignalFormatError(ValueError):
    """Raised when an upload cannot be read as a single 1-D signal."""


def _squeeze_to_1d(arr: np.ndarray, source: str) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.dtype.kind not in "fiub":
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise SignalFormatError(
                f"{source} contains non-numeric values. A signal must be numbers only."
            ) from exc

    squeezed = np.squeeze(arr)

    if squeezed.ndim == 0:
        raise SignalFormatError(f"{source} holds a single number, not a signal.")

    if squeezed.ndim == 2:
        rows, cols = squeezed.shape
        if cols == 1:
            squeezed = squeezed[:, 0]
        elif rows == 1:
            squeezed = squeezed[0, :]
        else:
            raise SignalFormatError(
                f"{source} is {rows}x{cols}. Upload one signal at a time: a single "
                f"column, a single row, or a flat array."
            )

    if squeezed.ndim != 1:
        raise SignalFormatError(
            f"{source} has shape {arr.shape}. Expected a 1-D signal of "
            f"{config.SIGNAL_LENGTH} points."
        )

    return squeezed.astype(np.float32, copy=False)


def _read_csv(data: bytes, source: str) -> np.ndarray:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        frame = pd.read_csv(io.StringIO(text), header=None)
    except Exception as exc:
        raise SignalFormatError(f"{source} could not be parsed as CSV: {exc}") from exc

    # A header row leaves a first row that is not numeric — drop it and retry.
    if frame.shape[0] > 1 and not np.issubdtype(frame.dtypes.iloc[0], np.number):
        frame = pd.read_csv(io.StringIO(text))

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if numeric.empty:
        raise SignalFormatError(f"{source} has no numeric data.")

    if numeric.isna().to_numpy().any():
        raise SignalFormatError(
            f"{source} has blank or non-numeric cells. Every value must be a number."
        )

    return _squeeze_to_1d(numeric.to_numpy(), source)


def _read_pickle(data: bytes, source: str) -> np.ndarray:
    try:
        obj = pickle.loads(data)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise SignalFormatError(
            f"'{source}' contains legacy pandas objects or metadata that cannot be unpickled ({exc}). "
            f"Suggestion: Export your 1-D EMG voltage signal as a standard .npy file (e.g. np.save('signal.npy', array)) or single-column .csv."
        ) from exc
    except Exception as exc:
        raise SignalFormatError(
            f"'{source}' could not be unpickled ({exc}). Suggestion: Please upload a valid .npy array or single-column .csv file."
        ) from exc

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = obj.to_numpy()
    elif isinstance(obj, (list, tuple)):
        obj = np.asarray(obj)

    if not isinstance(obj, np.ndarray):
        raise SignalFormatError(
            f"'{source}' contains a {type(obj).__name__}, not a numerical EMG recording. Expected a 1-D numerical array."
        )

    if obj.ndim == 3 and obj.shape[0] > 1:
        raise SignalFormatError(
            f"'{source}' contains multiple signals (shape {obj.shape}). Please upload one 1-D recording at a time."
        )

    return _squeeze_to_1d(obj, source)


def read_signal_bytes(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise SignalFormatError(
            f"Unsupported file type '{suffix or filename}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if suffix == ".npy":
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as exc:
            raise SignalFormatError(f"{filename} could not be read as .npy: {exc}") from exc
        return _squeeze_to_1d(arr, filename)

    if suffix in {".csv", ".txt"}:
        return _read_csv(data, filename)

    return _read_pickle(data, filename)


def read_signal_path(path: Path) -> np.ndarray:
    return read_signal_bytes(path.read_bytes(), path.name)


def validate_length(signal: np.ndarray) -> np.ndarray:
    expected = config.SIGNAL_LENGTH
    got = int(signal.shape[0])
    tol = config.LENGTH_TOLERANCE

    if got == expected:
        return signal

    if tol > 0 and abs(got - expected) <= tol:
        if got > expected:
            return signal[:expected]
        return np.pad(signal, (0, expected - got), mode="edge")

    raise SignalFormatError(
        f"Signal has {got:,} points but the CNN was trained on {expected:,}. "
        f"Resample or trim the recording to {expected:,} points, or set "
        f"SIGNAL_LENGTH in .env if your models expect a different length."
    )


def list_samples() -> list[dict]:
    if not config.SAMPLE_DIR.exists():
        return []

    out = []
    for path in sorted(config.SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        out.append(
            {
                "name": path.name,
                "label": path.stem.replace("_", " "),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return out


def resolve_sample(name: str) -> Path:
    """Resolve a sample name without letting it escape SAMPLE_DIR."""
    candidate = (config.SAMPLE_DIR / Path(name).name).resolve()
    root = config.SAMPLE_DIR.resolve()

    if root not in candidate.parents or not candidate.is_file():
        raise SignalFormatError(f"No bundled sample named '{name}'.")

    return candidate


def iter_supported(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p
