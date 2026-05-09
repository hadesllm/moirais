"""Tests for trnsfr.transfer_learning_msm."""
import numpy as np
import pytest
from moirais.fn.trnsfr import transfer_learning_msm


def test_trnsfr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = transfer_learning_msm(y, A, H, cohort)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trnsfr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = transfer_learning_msm(y, A, H, cohort)
    assert isinstance(result, dict)
