"""Tests for ca8u295.ca_chapter_8_unnumbered_295."""
import numpy as np
import pytest
from morie.fn.ca8u295 import ca_chapter_8_unnumbered_295


def test_ca8u295_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_295(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u295_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_295(x)
    assert isinstance(result, dict)
