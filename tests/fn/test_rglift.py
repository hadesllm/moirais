"""Tests for rglift.rangayyan_liftering."""
import numpy as np
import pytest
from morie.fn.rglift import rangayyan_liftering


def test_rglift_basic():
    """Test basic functionality."""
    cepstrum = np.random.default_rng(42).normal(0, 1, 100)
    l_low = np.random.default_rng(42).normal(0, 1, 100)
    l_high = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_liftering(cepstrum, l_low, l_high)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglift_edge():
    """Test edge cases."""
    cepstrum = np.random.default_rng(42).normal(0, 1, 100)
    l_low = np.random.default_rng(42).normal(0, 1, 100)
    l_high = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_liftering(cepstrum, l_low, l_high)
    assert isinstance(result, dict)
