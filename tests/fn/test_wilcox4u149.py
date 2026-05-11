"""Tests for wilcox4u149.wilcox_chapter_4_unnumbered_149."""
import numpy as np
import pytest
from morie.fn.wilcox4u149 import wilcox_chapter_4_unnumbered_149


def test_wilcox4u149_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_149(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u149_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_149(x)
    assert isinstance(result, dict)
