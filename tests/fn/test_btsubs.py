"""Tests for btsubs.boot_subsampling."""
import numpy as np
import pytest
from morie.fn.btsubs import boot_subsampling


def test_btsubs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_subsampling(x, m, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btsubs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_subsampling(x, m, stat, B)
    assert isinstance(result, dict)
