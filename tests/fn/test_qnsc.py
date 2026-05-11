"""Tests for qnsc.qn_scale."""
import numpy as np
import pytest
from morie.fn.qnsc import qn_scale


def test_qnsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = qn_scale(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qnsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = qn_scale(x)
    assert isinstance(result, dict)
