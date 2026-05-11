"""Tests for istfa.py - inverse STFT synthesis."""
import numpy as np
import pytest
from morie.fn.istfa import istft_synth, istfa


def test_istft_returns_signal_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = istft_synth(x)
    assert result.name == "istft"
    assert result.filtered is not None
    assert result.n_samples > 0


def test_istft_output_is_array():
    x = np.random.default_rng(42).standard_normal(256)
    result = istft_synth(x)
    assert isinstance(result.filtered, np.ndarray)


def test_istft_roundtrip_finite():
    x = np.random.default_rng(42).standard_normal(512)
    result = istft_synth(x, window_size=64, hop=32)
    assert np.all(np.isfinite(result.filtered))


def test_istfa_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = istfa(x)
    assert result.name == "istft"
