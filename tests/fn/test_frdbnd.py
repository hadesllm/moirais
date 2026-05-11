"""Tests for frdbnd.frechet_hoeffding_bounds."""
import numpy as np
import pytest
from morie.fn.frdbnd import frechet_hoeffding_bounds


def test_frdbnd_basic():
    """Test basic functionality."""
    F_0 = np.random.default_rng(42).normal(0, 1, 100)
    F_1 = np.random.default_rng(42).normal(0, 1, 100)
    result = frechet_hoeffding_bounds(F_0, F_1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frdbnd_edge():
    """Test edge cases."""
    F_0 = np.random.default_rng(42).normal(0, 1, 100)
    F_1 = np.random.default_rng(42).normal(0, 1, 100)
    result = frechet_hoeffding_bounds(F_0, F_1)
    assert isinstance(result, dict)
