"""Tests for lcfsdq.lc_first_sd_query."""
import numpy as np
import pytest
from moirais.fn.lcfsdq import lc_first_sd_query


def test_lcfsdq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = lc_first_sd_query(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lcfsdq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = lc_first_sd_query(x, coords)
    assert isinstance(result, dict)
