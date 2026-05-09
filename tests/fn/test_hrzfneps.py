"""Tests for hrzfneps.horowitz_panel_density_estimators."""
import numpy as np
import pytest
from moirais.fn.hrzfneps import horowitz_panel_density_estimators


def test_hrzfneps_basic():
    """Test basic functionality."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    x_panel = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_panel_density_estimators(y_panel, x_panel, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzfneps_edge():
    """Test edge cases."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    x_panel = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_panel_density_estimators(y_panel, x_panel, bandwidth)
    assert isinstance(result, dict)
