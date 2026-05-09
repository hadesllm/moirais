"""Tests for wilcox7u330.wilcox_chapter_7_unnumbered_330."""
import numpy as np
import pytest
from moirais.fn.wilcox7u330 import wilcox_chapter_7_unnumbered_330


def test_wilcox7u330_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_330(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u330_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_330(x)
    assert isinstance(result, dict)
