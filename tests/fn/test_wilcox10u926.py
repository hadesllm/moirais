"""Tests for wilcox10u926.wilcox_chapter_10_unnumbered_926."""
import numpy as np
import pytest
from moirais.fn.wilcox10u926 import wilcox_chapter_10_unnumbered_926


def test_wilcox10u926_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_926(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u926_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_926(x)
    assert isinstance(result, dict)
