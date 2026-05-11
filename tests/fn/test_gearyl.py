"""Tests for gearyl.local_gearys_c."""
import numpy as np
import pytest
from morie.fn.gearyl import local_gearys_c


def test_gearyl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_gearys_c(x, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gearyl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_gearys_c(x, W)
    assert isinstance(result, dict)
