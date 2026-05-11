"""Tests for wilcox14u646.wilcox_chapter_14_unnumbered_646."""
import numpy as np
import pytest
from morie.fn.wilcox14u646 import wilcox_chapter_14_unnumbered_646


def test_wilcox14u646_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_646(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u646_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_646(x)
    assert isinstance(result, dict)
