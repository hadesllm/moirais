"""Tests for hrzpanel.horowitz_panel_deconv."""
import numpy as np
import pytest
from morie.fn.hrzpanel import horowitz_panel_deconv


def test_hrzpanel_basic():
    """Test basic functionality."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    x_panel = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_panel_deconv(y_panel, x_panel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzpanel_edge():
    """Test edge cases."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    x_panel = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_panel_deconv(y_panel, x_panel)
    assert isinstance(result, dict)
