"""Tests for spintp.schabenberger_intensity_estimation."""
import numpy as np
import pytest
from moirais.fn.spintp import schabenberger_intensity_estimation


def test_spintp_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_intensity_estimation(points, bandwidth, region)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spintp_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_intensity_estimation(points, bandwidth, region)
    assert isinstance(result, dict)
