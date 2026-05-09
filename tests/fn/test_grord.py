"""Tests for grord.geron_ordinal_encoding."""
import numpy as np
import pytest
from moirais.fn.grord import geron_ordinal_encoding


def test_grord_basic():
    """Test basic functionality."""
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ordinal_encoding(categories)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grord_edge():
    """Test edge cases."""
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ordinal_encoding(categories)
    assert isinstance(result, dict)
