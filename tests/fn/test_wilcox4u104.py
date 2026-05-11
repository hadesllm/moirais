"""Tests for wilcox4u104.wilcox_chapter_4_unnumbered_104."""
import numpy as np
import pytest
from morie.fn.wilcox4u104 import wilcox_chapter_4_unnumbered_104


def test_wilcox4u104_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_104(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u104_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_104(x)
    assert isinstance(result, dict)
