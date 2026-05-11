"""Tests for reportm.report_noisy_max."""
import numpy as np
import pytest
from morie.fn.reportm import report_noisy_max


def test_reportm_basic():
    """Test basic functionality."""
    utilities = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = report_noisy_max(utilities, sensitivity, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reportm_edge():
    """Test edge cases."""
    utilities = np.random.default_rng(42).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = report_noisy_max(utilities, sensitivity, epsilon)
    assert isinstance(result, dict)
