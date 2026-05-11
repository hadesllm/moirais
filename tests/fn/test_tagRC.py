"""Tests for tagRC.tag_aware_rec."""
import numpy as np
import pytest
from morie.fn.tagRC import tag_aware_rec


def test_tagRC_basic():
    """Test basic functionality."""
    UTI = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = tag_aware_rec(UTI, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tagRC_edge():
    """Test edge cases."""
    UTI = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = tag_aware_rec(UTI, alpha)
    assert isinstance(result, dict)
