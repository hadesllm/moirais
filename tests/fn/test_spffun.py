"""Tests for spffun.schabenberger_f_function."""
import numpy as np
import pytest
from morie.fn.spffun import schabenberger_f_function


def test_spffun_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    r = 10
    result = schabenberger_f_function(points, region, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spffun_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    r = 10
    result = schabenberger_f_function(points, region, r)
    assert isinstance(result, dict)
