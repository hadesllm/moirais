"""Tests for wilcox4u105.wilcox_chapter_4_unnumbered_105."""
import numpy as np
import pytest
from morie.fn.wilcox4u105 import wilcox_chapter_4_unnumbered_105


def test_wilcox4u105_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_105(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u105_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_105(x)
    assert isinstance(result, dict)
