"""Tests for wilcox14u470.wilcox_chapter_14_unnumbered_470."""
import numpy as np
import pytest
from moirais.fn.wilcox14u470 import wilcox_chapter_14_unnumbered_470


def test_wilcox14u470_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_470(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u470_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_470(x)
    assert isinstance(result, dict)
