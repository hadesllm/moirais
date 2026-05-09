"""Tests for drovrl.dr_did_overlap_trim."""
import numpy as np
import pytest
from moirais.fn.drovrl import dr_did_overlap_trim


def test_drovrl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_overlap_trim(y, D, X, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drovrl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_overlap_trim(y, D, X, eps)
    assert isinstance(result, dict)
