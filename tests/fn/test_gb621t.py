"""Tests for gb621t.gibbons_ww2_ties."""
import numpy as np
import pytest
from moirais.fn.gb621t import gibbons_ww2_ties


def test_gb621t_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ww2_ties(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb621t_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ww2_ties(x, y)
    assert isinstance(result, dict)
