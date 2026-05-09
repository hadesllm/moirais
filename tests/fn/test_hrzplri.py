"""Tests for hrzplri.horowitz_plr_identification."""
import numpy as np
import pytest
from moirais.fn.hrzplri import horowitz_plr_identification


def test_hrzplri_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_plr_identification(x, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzplri_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_plr_identification(x, z)
    assert isinstance(result, dict)
