"""Tests for medmsm.msm_mediation."""
import numpy as np
import pytest
from morie.fn.medmsm import msm_mediation


def test_medmsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_mediation(y, A, M, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medmsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_mediation(y, A, M, H)
    assert isinstance(result, dict)
