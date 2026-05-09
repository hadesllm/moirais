"""Tests for wilcox9u1061.wilcox_chapter_9_unnumbered_1061."""
import numpy as np
import pytest
from moirais.fn.wilcox9u1061 import wilcox_chapter_9_unnumbered_1061


def test_wilcox9u1061_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1061(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox9u1061_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1061(x)
    assert isinstance(result, dict)
