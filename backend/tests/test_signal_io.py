"""
Loader tests for signal_io.

Every rejection path must raise SignalFormatError with a message that says what
was expected -- main.py turns those into a 400 the clinician can act on, so a
bare ValueError or a stack trace here would leak as a 500.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("ENABLE_FLORENCE", "0")
os.environ.setdefault("ENABLE_LLM", "0")
os.environ.setdefault("SERVE_FRONTEND", "0")

import config
import signal_io
from signal_io import SignalFormatError


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def csv_bytes(array: np.ndarray, header: str | None = None) -> bytes:
    lines = [header] if header else []
    lines += [repr(float(v)) for v in array]
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def signal() -> np.ndarray:
    return np.random.default_rng(1).standard_normal(config.SIGNAL_LENGTH).astype(np.float32)


# --- .npy -------------------------------------------------------------------
def test_npy_round_trip(signal):
    out = signal_io.read_signal_bytes(npy_bytes(signal), "signal.npy")
    assert out.shape == (config.SIGNAL_LENGTH,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, signal, rtol=1e-6)


def test_npy_column_vector_is_squeezed(signal):
    out = signal_io.read_signal_bytes(npy_bytes(signal.reshape(-1, 1)), "col.npy")
    assert out.shape == (config.SIGNAL_LENGTH,)


def test_npy_row_vector_is_squeezed(signal):
    out = signal_io.read_signal_bytes(npy_bytes(signal.reshape(1, -1)), "row.npy")
    assert out.shape == (config.SIGNAL_LENGTH,)


def test_npy_with_many_signals_is_rejected():
    batch = np.zeros((4, config.SIGNAL_LENGTH), dtype=np.float32)
    with pytest.raises(SignalFormatError, match="one signal at a time"):
        signal_io.read_signal_bytes(npy_bytes(batch), "batch.npy")


def test_npy_scalar_is_rejected():
    with pytest.raises(SignalFormatError, match="single number"):
        signal_io.read_signal_bytes(npy_bytes(np.array(3.0)), "scalar.npy")


def test_corrupt_npy_is_a_format_error():
    with pytest.raises(SignalFormatError, match=r"could not be read as \.npy"):
        signal_io.read_signal_bytes(b"not a numpy file at all", "broken.npy")


def test_pickled_npy_is_refused():
    # allow_pickle=False is what stops an upload from executing code on load.
    payload = npy_bytes(np.array([{"a": 1}], dtype=object))
    with pytest.raises(SignalFormatError):
        signal_io.read_signal_bytes(payload, "evil.npy")


# --- .csv -------------------------------------------------------------------
def test_csv_single_column_round_trip(signal):
    out = signal_io.read_signal_bytes(csv_bytes(signal), "signal.csv")
    assert out.shape == (config.SIGNAL_LENGTH,)
    np.testing.assert_allclose(out, signal, rtol=1e-5)


def test_csv_with_a_header_row_is_handled(signal):
    out = signal_io.read_signal_bytes(csv_bytes(signal, header="amplitude"), "headed.csv")
    assert out.shape == (config.SIGNAL_LENGTH,)


def test_csv_with_a_blank_cell_is_rejected():
    payload = b"1.0\n2.0\n\n4.0\n,\n"
    with pytest.raises(SignalFormatError):
        signal_io.read_signal_bytes(payload, "gappy.csv")


def test_csv_with_no_numbers_is_rejected():
    with pytest.raises(SignalFormatError):
        signal_io.read_signal_bytes(b"alpha\nbeta\ngamma\n", "words.csv")


def test_txt_uses_the_csv_reader(signal):
    out = signal_io.read_signal_bytes(csv_bytes(signal[:100]), "signal.txt")
    assert out.shape == (100,)


# --- extensions -------------------------------------------------------------
def test_unsupported_extension_names_what_is_supported():
    with pytest.raises(SignalFormatError, match="Unsupported file type"):
        signal_io.read_signal_bytes(b"\x00\x01", "recording.edf")


# --- validate_length --------------------------------------------------------
def test_exact_length_passes(signal):
    assert signal_io.validate_length(signal).shape[0] == config.SIGNAL_LENGTH


def test_wrong_length_names_both_numbers():
    short = np.zeros(1000, dtype=np.float32)
    with pytest.raises(SignalFormatError) as excinfo:
        signal_io.validate_length(short)

    message = str(excinfo.value)
    assert "1,000" in message
    assert f"{config.SIGNAL_LENGTH:,}" in message


def test_a_too_long_signal_is_trimmed_within_tolerance(monkeypatch):
    monkeypatch.setattr(config, "LENGTH_TOLERANCE", 50)
    out = signal_io.validate_length(np.zeros(config.SIGNAL_LENGTH + 10, dtype=np.float32))
    assert out.shape[0] == config.SIGNAL_LENGTH


def test_a_too_short_signal_is_padded_within_tolerance(monkeypatch):
    monkeypatch.setattr(config, "LENGTH_TOLERANCE", 50)
    out = signal_io.validate_length(np.ones(config.SIGNAL_LENGTH - 10, dtype=np.float32))
    assert out.shape[0] == config.SIGNAL_LENGTH
    assert out[-1] == pytest.approx(1.0)  # padded with the edge value


def test_tolerance_does_not_rescue_a_wildly_wrong_length(monkeypatch):
    monkeypatch.setattr(config, "LENGTH_TOLERANCE", 50)
    with pytest.raises(SignalFormatError):
        signal_io.validate_length(np.zeros(500, dtype=np.float32))


# --- samples ----------------------------------------------------------------
def test_resolve_sample_blocks_path_traversal():
    with pytest.raises(SignalFormatError):
        signal_io.resolve_sample("../../etc/passwd")


def test_resolve_sample_rejects_an_unknown_name():
    with pytest.raises(SignalFormatError, match="No bundled sample"):
        signal_io.resolve_sample("does_not_exist.npy")
