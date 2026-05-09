"""Tests for hrzpanms.horowitz_panel_max_score."""
import numpy as np
import pytest
from moirais.fn.hrzpanms import horowitz_panel_max_score


def test_hrzpanms_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_periods = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_panel_max_score(x, y, n_periods)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzpanms_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_periods = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_panel_max_score(x, y, n_periods)
    assert isinstance(result, dict)
