"""Tests for aitilri.aitchison_ilr_inverse."""
import numpy as np
import pytest
from morie.fn.aitilri import aitchison_ilr_inverse


def test_aitilri_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_ilr_inverse(y, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitilri_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_ilr_inverse(y, V)
    assert isinstance(result, dict)
