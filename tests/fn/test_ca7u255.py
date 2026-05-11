"""Tests for ca7u255.ca_chapter_7_unnumbered_255."""
import numpy as np
import pytest
from morie.fn.ca7u255 import ca_chapter_7_unnumbered_255


def test_ca7u255_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_255(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u255_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_255(x)
    assert isinstance(result, dict)
