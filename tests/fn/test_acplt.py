"""Tests for moirais.fn.acplt -- Autocorrelation data."""

import numpy as np
from moirais.fn.acplt import autocorrelation_data


def test_returns_dict():
    samples = np.random.default_rng(42).standard_normal(500)
    result = autocorrelation_data(samples)
    assert isinstance(result, dict)
    assert "acf" in result


def test_acf_starts_at_one():
    samples = np.random.default_rng(42).standard_normal(500)
    result = autocorrelation_data(samples)
    assert result["acf"][0] == 1.0


def test_acf_length():
    samples = np.random.default_rng(42).standard_normal(500)
    result = autocorrelation_data(samples, max_lag=30)
    assert len(result["acf"]) == 31
