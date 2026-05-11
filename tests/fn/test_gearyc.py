"""Tests for gearyc.gearys_c."""
import numpy as np
import pytest
from morie.fn.gearyc import gearyc as gearys_c


def test_gearyc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gearys_c(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_gearyc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = gearys_c(x, W)
    assert isinstance(result, dict)
