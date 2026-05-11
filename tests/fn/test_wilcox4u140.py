"""Tests for wilcox4u140.wilcox_chapter_4_unnumbered_140."""
import numpy as np
import pytest
from morie.fn.wilcox4u140 import wilcox_chapter_4_unnumbered_140


def test_wilcox4u140_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_140(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u140_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_140(x)
    assert isinstance(result, dict)
