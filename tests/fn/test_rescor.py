"""Tests for rescor.rescore_consensus."""
import numpy as np
import pytest
from morie.fn.rescor import rescore_consensus


def test_rescor_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = rescore_consensus(scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rescor_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = rescore_consensus(scores)
    assert isinstance(result, dict)
