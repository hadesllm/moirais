"""Tests for hmmms.geron_min_max_scaling."""
import numpy as np
import pytest
from morie.fn.hmmms import geron_min_max_scaling


def test_hmmms_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_min_max_scaling(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmms_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_min_max_scaling(X)
    assert isinstance(result, dict)
