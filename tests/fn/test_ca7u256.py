"""Tests for ca7u256.ca_chapter_7_unnumbered_256."""
import numpy as np
import pytest
from morie.fn.ca7u256 import ca_chapter_7_unnumbered_256


def test_ca7u256_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_256(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u256_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_256(x)
    assert isinstance(result, dict)
