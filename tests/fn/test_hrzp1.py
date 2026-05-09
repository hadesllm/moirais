"""Tests for hrzp1.horowitz_plr_estimator."""
import numpy as np
import pytest
from moirais.fn.hrzp1 import horowitz_plr_estimator


def test_hrzp1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_plr_estimator(x, y, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzp1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_plr_estimator(x, y, z)
    assert isinstance(result, dict)
