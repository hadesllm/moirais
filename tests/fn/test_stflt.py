"""Tests for stflt.py - Subband filter."""
import numpy as np
from morie.fn.stflt import subband_filter, stflt


def test_subband_returns_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = subband_filter(x, level=3, band=0)
    assert result.name == "subband_filter"
    assert "subband" in result.extra


def test_subband_approx_label():
    x = np.random.default_rng(42).standard_normal(128)
    result = subband_filter(x, level=3, band=0)
    assert result.extra["label"].startswith("A")


def test_subband_detail_band():
    x = np.random.default_rng(42).standard_normal(256)
    result = subband_filter(x, level=3, band=1)
    assert "subband" in result.extra


def test_subband_invalid_band():
    import pytest
    x = np.random.default_rng(42).standard_normal(128)
    with pytest.raises(ValueError):
        subband_filter(x, level=2, band=99)


def test_subband_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = stflt(x, level=2, band=0)
    assert result.name == "subband_filter"
