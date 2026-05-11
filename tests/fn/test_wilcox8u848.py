"""Tests for wilcox8u848.wilcox_chapter_8_unnumbered_848."""
import numpy as np
import pytest
from morie.fn.wilcox8u848 import wilcox_chapter_8_unnumbered_848


def test_wilcox8u848_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_848(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u848_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_848(x)
    assert isinstance(result, dict)
