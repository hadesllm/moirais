"""Tests for rng229.rangayyan_ch4_matched_filter_output_psd."""
import numpy as np
import pytest
from morie.fn.rng229 import rangayyan_ch4_matched_filter_output_psd


def test_rng229_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_output_psd(X, H, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng229_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_output_psd(X, H, f)
    assert isinstance(result, dict)
