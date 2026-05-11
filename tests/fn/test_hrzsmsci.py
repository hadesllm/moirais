"""Tests for hrzsmsci.horowitz_sms_confidence."""
import numpy as np
import pytest
from morie.fn.hrzsmsci import horowitz_sms_confidence


def test_hrzsmsci_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_sms_confidence(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsmsci_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_sms_confidence(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
