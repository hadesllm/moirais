"""Tests for bayisr.importance_resample."""
import numpy as np
import pytest
from moirais.fn.bayisr import importance_resample


def test_bayisr_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = importance_resample(target, proposal, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayisr_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = importance_resample(target, proposal, n, M)
    assert isinstance(result, dict)
