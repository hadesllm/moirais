"""Tests for regmlm.regenie_lmm."""
import numpy as np
import pytest
from morie.fn.regmlm import regenie_lmm


def test_regmlm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = regenie_lmm(y, M, blocks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_regmlm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = regenie_lmm(y, M, blocks)
    assert isinstance(result, dict)
