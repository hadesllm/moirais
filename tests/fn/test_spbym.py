"""Tests for spbym.schabenberger_bym_model."""
import numpy as np
import pytest
from morie.fn.spbym import schabenberger_bym_model


def test_spbym_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_bym_model(x, y, E, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spbym_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_bym_model(x, y, E, w)
    assert isinstance(result, dict)
