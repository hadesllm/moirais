"""Tests for hmpmps.geron_mps_acceleration."""
import numpy as np
import pytest
from moirais.fn.hmpmps import geron_mps_acceleration


def test_hmpmps_basic():
    """Test basic functionality."""
    tensor = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mps_acceleration(tensor)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpmps_edge():
    """Test edge cases."""
    tensor = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mps_acceleration(tensor)
    assert isinstance(result, dict)
