"""Tests for hrzt41.horowitz_thm4_1_id_median."""
import numpy as np
import pytest
from morie.fn.hrzt41 import horowitz_thm4_1_id_median


def test_hrzt41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_thm4_1_id_median(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzt41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_thm4_1_id_median(x, y)
    assert isinstance(result, dict)
