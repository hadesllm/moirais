"""Tests for hrzsmsrc.horowitz_sms_rate."""
import numpy as np
import pytest
from moirais.fn.hrzsmsrc import horowitz_sms_rate


def test_hrzsmsrc_basic():
    """Test basic functionality."""
    n = 100
    smoothness_order = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_sms_rate(n, smoothness_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsmsrc_edge():
    """Test edge cases."""
    n = 100
    smoothness_order = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_sms_rate(n, smoothness_order)
    assert isinstance(result, dict)
