"""Tests for tukrho.tukey_biweight."""
import numpy as np
import pytest
from morie.fn.tukrho import tukey_biweight


def test_tukrho_basic():
    """Test basic functionality."""
    r = 10
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_biweight(r, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tukrho_edge():
    """Test edge cases."""
    r = 10
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_biweight(r, c)
    assert isinstance(result, dict)
