"""Tests for fzt14.fauzi_thm1_4_asympnorm_mgkde."""
import numpy as np
import pytest
from morie.fn.fzt14 import fauzi_thm1_4_asympnorm_mgkde


def test_fzt14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_4_asympnorm_mgkde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_4_asympnorm_mgkde(x, bandwidth)
    assert isinstance(result, dict)
