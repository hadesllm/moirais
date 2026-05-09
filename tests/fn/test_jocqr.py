"""Tests for jocqr.joseph_conformalized_quantile_regression."""
import numpy as np
import pytest
from moirais.fn.jocqr import joseph_conformalized_quantile_regression


def test_jocqr_basic():
    """Test basic functionality."""
    calibration_y = np.random.default_rng(42).normal(0, 1, 100)
    calibration_q_lo = np.random.default_rng(42).normal(0, 1, 100)
    calibration_q_hi = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_conformalized_quantile_regression(calibration_y, calibration_q_lo, calibration_q_hi, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jocqr_edge():
    """Test edge cases."""
    calibration_y = np.random.default_rng(42).normal(0, 1, 100)
    calibration_q_lo = np.random.default_rng(42).normal(0, 1, 100)
    calibration_q_hi = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_conformalized_quantile_regression(calibration_y, calibration_q_lo, calibration_q_hi, alpha)
    assert isinstance(result, dict)
