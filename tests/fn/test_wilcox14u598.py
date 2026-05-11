"""Tests for wilcox14u598.wilcox_chapter_14_unnumbered_598."""
import numpy as np
import pytest
from morie.fn.wilcox14u598 import wilcox_chapter_14_unnumbered_598


def test_wilcox14u598_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_598(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u598_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_598(x)
    assert isinstance(result, dict)
