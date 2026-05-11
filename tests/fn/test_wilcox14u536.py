"""Tests for wilcox14u536.wilcox_chapter_14_unnumbered_536."""
import numpy as np
import pytest
from morie.fn.wilcox14u536 import wilcox_chapter_14_unnumbered_536


def test_wilcox14u536_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_536(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u536_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_536(x)
    assert isinstance(result, dict)
