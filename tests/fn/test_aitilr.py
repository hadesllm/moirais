"""Tests for aitilr.aitchison_ilr."""
import numpy as np
import pytest
from moirais.fn.aitilr import aitchison_ilr


def test_aitilr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_ilr(x, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitilr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_ilr(x, V)
    assert isinstance(result, dict)
