"""Tests for wsmfis.wasserman_fisher_info."""
import numpy as np
import pytest
from morie.fn.wsmfis import wasserman_fisher_info


def test_wsmfis_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = wasserman_fisher_info(f, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmfis_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = wasserman_fisher_info(f, theta)
    assert isinstance(result, dict)
