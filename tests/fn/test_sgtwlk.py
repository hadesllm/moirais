"""Tests for sgtwlk.sgt_weisfeiler_leman_relabel."""
import numpy as np
import pytest
from morie.fn.sgtwlk import sgt_weisfeiler_leman_relabel


def test_sgtwlk_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels0 = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_weisfeiler_leman_relabel(A, labels0, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtwlk_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels0 = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_weisfeiler_leman_relabel(A, labels0, max_iter)
    assert isinstance(result, dict)
