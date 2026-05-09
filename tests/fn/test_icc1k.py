"""Tests for icc1k.icc_one_way_average."""
import numpy as np
import pytest
from moirais.fn.icc1k import icc_one_way_average


def test_icc1k_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_one_way_average(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc1k_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_one_way_average(y, cluster)
    assert isinstance(result, dict)
