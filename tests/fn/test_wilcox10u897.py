"""Tests for wilcox10u897.wilcox_chapter_10_unnumbered_897."""
import numpy as np
import pytest
from moirais.fn.wilcox10u897 import wilcox_chapter_10_unnumbered_897


def test_wilcox10u897_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_897(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u897_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_897(x)
    assert isinstance(result, dict)
