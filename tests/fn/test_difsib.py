"""Tests for difsib.dif_sibtest."""
import numpy as np
import pytest
from moirais.fn.difsib import dif_sibtest


def test_difsib_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    studied = np.random.default_rng(42).normal(0, 1, 100)
    matching = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_sibtest(y, group, studied, matching)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_difsib_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    studied = np.random.default_rng(42).normal(0, 1, 100)
    matching = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_sibtest(y, group, studied, matching)
    assert isinstance(result, dict)
