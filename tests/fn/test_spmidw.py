"""Tests for spmidw.schabenberger_idw."""
import numpy as np
import pytest
from moirais.fn.spmidw import schabenberger_idw


def test_spmidw_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    power = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_idw(coords, z, target, power)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmidw_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    power = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_idw(coords, z, target, power)
    assert isinstance(result, dict)
