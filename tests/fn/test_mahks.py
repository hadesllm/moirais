"""Tests for mahks.ma_hartung_knapp."""
import numpy as np
import pytest
from morie.fn.mahks import ma_hartung_knapp


def test_mahks_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_hartung_knapp(yi, vi, tau2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mahks_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_hartung_knapp(yi, vi, tau2)
    assert isinstance(result, dict)
