"""Tests for sgtsbms.sgt_sbm_spectral_estimate."""
import numpy as np
import pytest
from morie.fn.sgtsbms import sgt_sbm_spectral_estimate


def test_sgtsbms_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_sbm_spectral_estimate(A, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtsbms_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_sbm_spectral_estimate(A, K)
    assert isinstance(result, dict)
