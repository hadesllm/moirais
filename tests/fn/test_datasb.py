"""Tests for datasb.data_subset_refutation."""
import numpy as np
import pytest
from morie.fn.datasb import data_subset_refutation


def test_datasb_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    subset_fraction = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = data_subset_refutation(model, subset_fraction, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_datasb_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    subset_fraction = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = data_subset_refutation(model, subset_fraction, n_iter)
    assert isinstance(result, dict)
