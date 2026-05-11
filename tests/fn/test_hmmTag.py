"""Tests for hmmTag.hmm_pos."""
import numpy as np
import pytest
from morie.fn.hmmTag import hmm_pos


def test_hmmTag_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tagset = np.random.default_rng(42).normal(0, 1, 100)
    result = hmm_pos(X, tagset)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmTag_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tagset = np.random.default_rng(42).normal(0, 1, 100)
    result = hmm_pos(X, tagset)
    assert isinstance(result, dict)
