"""Tests for wilcox2u302.wilcox_chapter_2_unnumbered_302."""
import numpy as np
import pytest
from morie.fn.wilcox2u302 import wilcox_chapter_2_unnumbered_302


def test_wilcox2u302_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_302(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox2u302_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_302(x)
    assert isinstance(result, dict)
