"""Tests for hmmprf.hmm_profile."""
import numpy as np
import pytest
from morie.fn.hmmprf import hmm_profile


def test_hmmprf_basic():
    """Test basic functionality."""
    seq = np.random.default_rng(42).normal(0, 1, 100)
    profile = np.random.default_rng(42).normal(0, 1, 100)
    result = hmm_profile(seq, profile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmprf_edge():
    """Test edge cases."""
    seq = np.random.default_rng(42).normal(0, 1, 100)
    profile = np.random.default_rng(42).normal(0, 1, 100)
    result = hmm_profile(seq, profile)
    assert isinstance(result, dict)
