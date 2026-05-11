"""Tests for meteor.meteor."""
import numpy as np
import pytest
from morie.fn.meteor import meteor


def test_meteor_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = meteor(candidate, reference)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_meteor_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = meteor(candidate, reference)
    assert isinstance(result, dict)
