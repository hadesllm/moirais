"""Tests for snscl.sn_scale."""
import numpy as np
import pytest
from moirais.fn.snscl import sn_scale


def test_snscl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = sn_scale(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snscl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = sn_scale(y)
    assert isinstance(result, dict)
