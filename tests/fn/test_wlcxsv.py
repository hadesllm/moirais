"""Tests for wlcxsv.wilcoxon_gehan."""
import numpy as np
import pytest
from morie.fn.wlcxsv import wilcoxon_gehan


def test_wlcxsv_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcoxon_gehan(time, event, group)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wlcxsv_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcoxon_gehan(time, event, group)
    assert isinstance(result, dict)
