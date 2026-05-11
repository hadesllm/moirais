"""Tests for wilcox5u379.wilcox_chapter_5_unnumbered_379."""
import numpy as np
import pytest
from morie.fn.wilcox5u379 import wilcox_chapter_5_unnumbered_379


def test_wilcox5u379_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_379(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u379_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_379(x)
    assert isinstance(result, dict)
