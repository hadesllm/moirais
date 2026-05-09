"""Tests for mabaujat.ma_baujat_plot_data."""
import numpy as np
import pytest
from moirais.fn.mabaujat import ma_baujat_plot_data


def test_mabaujat_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_baujat_plot_data(yi, vi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mabaujat_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_baujat_plot_data(yi, vi)
    assert isinstance(result, dict)
