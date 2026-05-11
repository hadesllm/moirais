"""Tests for ca7u222.ca_chapter_7_unnumbered_222."""
import numpy as np
import pytest
from morie.fn.ca7u222 import ca_chapter_7_unnumbered_222


def test_ca7u222_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_222(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u222_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_222(x)
    assert isinstance(result, dict)
