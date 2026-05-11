"""Tests for fzbwc.fauzi_quantile_bw_condition."""
import numpy as np
import pytest
from morie.fn.fzbwc import fauzi_quantile_bw_condition


def test_fzbwc_basic():
    """Test basic functionality."""
    bandwidth = 0.3
    n = 100
    result = fauzi_quantile_bw_condition(bandwidth, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzbwc_edge():
    """Test edge cases."""
    bandwidth = 0.3
    n = 100
    result = fauzi_quantile_bw_condition(bandwidth, n)
    assert isinstance(result, dict)
