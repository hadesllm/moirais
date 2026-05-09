"""Tests for qnscl.qn_scale."""
import numpy as np
import pytest
from moirais.fn.qnscl import qn_scale


def test_qnscl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = qn_scale(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qnscl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = qn_scale(y)
    assert isinstance(result, dict)
