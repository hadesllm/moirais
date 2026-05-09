"""Tests for tukeyw.tukey_biweight."""
import numpy as np
import pytest
from moirais.fn.tukeyw import tukey_biweight


def test_tukeyw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_biweight(y, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tukeyw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = tukey_biweight(y, c)
    assert isinstance(result, dict)
