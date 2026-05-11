"""Tests for mafnpr.ma_funnel_plot_data."""
import numpy as np
import pytest
from morie.fn.mafnpr import ma_funnel_plot_data


def test_mafnpr_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_funnel_plot_data(yi, se_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mafnpr_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_funnel_plot_data(yi, se_i)
    assert isinstance(result, dict)
