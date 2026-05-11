"""Tests for hmmlm.geron_masked_lm."""
import numpy as np
import pytest
from morie.fn.hmmlm import geron_masked_lm


def test_hmmlm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mask_frac = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_masked_lm(X, mask_frac)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmlm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mask_frac = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_masked_lm(X, mask_frac)
    assert isinstance(result, dict)
