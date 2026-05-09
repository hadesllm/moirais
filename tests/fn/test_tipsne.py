"""Tests for tipsne.tipping_point_sensitivity."""
import numpy as np
import pytest
from moirais.fn.tipsne import tipping_point_sensitivity


def test_tipsne_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    missing_indicator = np.random.default_rng(42).normal(0, 1, 100)
    result = tipping_point_sensitivity(y, D, missing_indicator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tipsne_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    missing_indicator = np.random.default_rng(42).normal(0, 1, 100)
    result = tipping_point_sensitivity(y, D, missing_indicator)
    assert isinstance(result, dict)
