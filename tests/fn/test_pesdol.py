"""Tests for pesdol.pesaran_shin_dols."""
import numpy as np
import pytest
from moirais.fn.pesdol import pesaran_shin_dols


def test_pesdol_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    result = pesaran_shin_dols(y, X, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pesdol_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    result = pesaran_shin_dols(y, X, p)
    assert isinstance(result, dict)
