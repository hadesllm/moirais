"""Tests for sqztf.py - Synchrosqueezed CWT."""
import numpy as np
from moirais.fn.sqztf import synchrosqueeze, sqztf


def test_sqztf_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = synchrosqueeze(x, fs=128.0)
    assert result.name == "synchrosqueeze"
    assert "ssq" in result.extra


def test_sqztf_ssq_shape():
    x = np.random.default_rng(42).standard_normal(64)
    scales = np.arange(1, 17, dtype=float)
    result = synchrosqueeze(x, scales=scales)
    assert result.extra["ssq"].shape[0] == len(scales)
    assert result.extra["ssq"].shape[1] == len(x)


def test_sqztf_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = sqztf(x)
    assert result.name == "synchrosqueeze"
