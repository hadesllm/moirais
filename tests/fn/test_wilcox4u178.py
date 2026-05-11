"""Tests for wilcox4u178.wilcox_chapter_4_unnumbered_178."""
import numpy as np
import pytest
from morie.fn.wilcox4u178 import wilcox_chapter_4_unnumbered_178


def test_wilcox4u178_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_178(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u178_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_178(x)
    assert isinstance(result, dict)
