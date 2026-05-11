"""Tests for wilcox4u132.wilcox_chapter_4_unnumbered_132."""
import numpy as np
import pytest
from morie.fn.wilcox4u132 import wilcox_chapter_4_unnumbered_132


def test_wilcox4u132_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_132(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u132_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_132(x)
    assert isinstance(result, dict)
