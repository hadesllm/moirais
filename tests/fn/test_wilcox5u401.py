"""Tests for wilcox5u401.wilcox_chapter_5_unnumbered_401."""
import numpy as np
import pytest
from morie.fn.wilcox5u401 import wilcox_chapter_5_unnumbered_401


def test_wilcox5u401_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_401(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u401_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_401(x)
    assert isinstance(result, dict)
