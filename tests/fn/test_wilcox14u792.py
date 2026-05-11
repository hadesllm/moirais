"""Tests for wilcox14u792.wilcox_chapter_14_unnumbered_792."""
import numpy as np
import pytest
from morie.fn.wilcox14u792 import wilcox_chapter_14_unnumbered_792


def test_wilcox14u792_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_792(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u792_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_792(x)
    assert isinstance(result, dict)
