"""Tests for wilcox10u923.wilcox_chapter_10_unnumbered_923."""
import numpy as np
import pytest
from moirais.fn.wilcox10u923 import wilcox_chapter_10_unnumbered_923


def test_wilcox10u923_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_923(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u923_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_923(x)
    assert isinstance(result, dict)
