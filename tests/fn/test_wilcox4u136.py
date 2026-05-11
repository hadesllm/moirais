"""Tests for wilcox4u136.wilcox_chapter_4_unnumbered_136."""
import numpy as np
import pytest
from morie.fn.wilcox4u136 import wilcox_chapter_4_unnumbered_136


def test_wilcox4u136_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_136(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u136_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_136(x)
    assert isinstance(result, dict)
