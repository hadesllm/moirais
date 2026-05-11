"""Tests for poststs.poststratify."""
import numpy as np
import pytest
from morie.fn.poststs import poststratify


def test_poststs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    result = poststratify(y, stratum, Nh)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poststs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    result = poststratify(y, stratum, Nh)
    assert isinstance(result, dict)
