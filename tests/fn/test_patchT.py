"""Tests for patchT.patch_tst."""
import numpy as np
import pytest
from morie.fn.patchT import patch_tst


def test_patchT_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    patch_len = np.random.default_rng(42).normal(0, 1, 100)
    result = patch_tst(X, patch_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_patchT_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    patch_len = np.random.default_rng(42).normal(0, 1, 100)
    result = patch_tst(X, patch_len)
    assert isinstance(result, dict)
