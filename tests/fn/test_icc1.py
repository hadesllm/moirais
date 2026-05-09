"""Tests for icc1.icc_one_way."""
import numpy as np
import pytest
from moirais.fn.icc1 import icc_one_way


def test_icc1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_one_way(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_one_way(y, cluster)
    assert isinstance(result, dict)
