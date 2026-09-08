"""
Preprocessing tests: the notebook-ported numeric steps in pipeline.py.

These lock down behaviour the trained artifacts depend on. If a test here fails,
the CNN and Florence-2 are being fed something they were not trained on, and the
reported accuracy no longer applies -- fix the code, not the test.
"""

from __future__ import annotations

import os
import sys
from itertools import pairwise
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
from pipeline import (
    bandpass_filter,
    find_abnormal_regions,
    preprocess_signal,
    segment_signal,
    zscore_normalize,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- bandpass_filter --------------------------------------------------------
def test_bandpass_keeps_length_and_shape(rng):
    sig = rng.standard_normal(4096).astype(np.float32)
    out = bandpass_filter(sig)
    assert out.shape == sig.shape


def test_bandpass_removes_dc_offset(rng):
    # A 5-450 Hz passband must reject a constant offset.
    sig = rng.standard_normal(4096) + 50.0
    out = bandpass_filter(sig)
    assert abs(float(np.mean(out))) < 1.0


def test_bandpass_keeps_an_in_band_tone_and_drops_an_out_of_band_one():
    t = np.arange(4096) / config.FS
    in_band = np.sin(2 * np.pi * 100 * t)  # 100 Hz, inside 5-450
    out_of_band = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, below the low cutoff

    kept = float(np.std(bandpass_filter(in_band)))
    dropped = float(np.std(bandpass_filter(out_of_band)))

    assert kept > 0.5 * float(np.std(in_band))
    assert dropped < 0.1 * float(np.std(out_of_band))


def test_bandpass_clamps_a_high_cutoff_above_nyquist():
    # high/nyq must be clamped below 1.0 or butter() raises.
    sig = np.sin(np.linspace(0, 50, 2048))
    out = bandpass_filter(sig, low=5.0, high=config.FS)  # far above Nyquist
    assert np.isfinite(out).all()


# --- zscore_normalize -------------------------------------------------------
def test_zscore_gives_zero_mean_unit_std(rng):
    sig = rng.standard_normal(2048) * 7.0 + 3.0
    out = zscore_normalize(sig)
    assert float(np.mean(out)) == pytest.approx(0.0, abs=1e-5)
    assert float(np.std(out)) == pytest.approx(1.0, abs=1e-4)


def test_zscore_survives_a_constant_signal():
    # std is 0 here; the 1e-8 epsilon is what keeps this finite.
    out = zscore_normalize(np.full(512, 4.2))
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 1e-3


def test_zscore_is_scale_invariant(rng):
    sig = rng.standard_normal(1024)
    np.testing.assert_allclose(zscore_normalize(sig), zscore_normalize(sig * 100.0), atol=1e-4)


# --- preprocess_signal ------------------------------------------------------
def test_preprocess_returns_a_flat_float_array_of_the_same_length(rng):
    sig = rng.standard_normal((config.SIGNAL_LENGTH, 1)).astype(np.float64)
    out = preprocess_signal(sig)
    assert out.ndim == 1
    assert out.shape[0] == config.SIGNAL_LENGTH
    assert np.isfinite(out).all()


def test_preprocess_falls_back_to_the_raw_signal_when_the_filter_fails():
    # Two samples is far too short for a 4th-order filtfilt; the notebook
    # swallows that and z-scores the raw signal instead.
    out = preprocess_signal(np.array([1.0, 5.0]))
    assert out.shape == (2,)
    assert np.isfinite(out).all()


# --- segment_signal ---------------------------------------------------------
def test_segments_tile_the_signal_without_gaps_or_overlap():
    segments = segment_signal(np.zeros(config.SIGNAL_LENGTH))

    assert len(segments) == config.N_SEGMENTS
    assert segments[0][0] == 0
    assert segments[-1][1] == config.SIGNAL_LENGTH
    for (_, prev_end), (next_start, _) in pairwise(segments):
        assert prev_end == next_start


def test_last_segment_absorbs_the_remainder():
    # 23437 / 10 does not divide evenly; the tail must not be dropped.
    segments = segment_signal(np.zeros(23437), n_segments=10)
    assert segments[-1][1] == 23437
    assert segments[-1][1] - segments[-1][0] == 2343 + 7


def test_segment_count_is_configurable():
    assert len(segment_signal(np.zeros(1000), n_segments=4)) == 4


# --- find_abnormal_regions --------------------------------------------------
def test_a_flat_signal_flags_no_abnormal_region():
    regions, scores = find_abnormal_regions(np.ones(1000))
    assert regions == []
    assert len(scores) == config.N_SEGMENTS


def test_a_single_high_energy_burst_is_flagged():
    sig = np.zeros(1000)
    sig[500:600] = 20.0  # sits inside segment 5
    regions, _scores = find_abnormal_regions(sig, n_segments=10)

    assert len(regions) == 1
    start, end = regions[0]
    assert start <= 500 < end
