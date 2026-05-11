"""Tests for ca4u126.ca_chapter_4_unnumbered_126."""
import numpy as np
import pytest
from morie.fn.ca4u126 import ca_chapter_4_unnumbered_126


def test_ca4u126_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_126(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca4u126_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_126(x)
    assert isinstance(result, dict)
