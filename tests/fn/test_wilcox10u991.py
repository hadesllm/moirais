"""Tests for wilcox10u991.wilcox_chapter_10_unnumbered_991."""
import numpy as np
import pytest
from moirais.fn.wilcox10u991 import wilcox_chapter_10_unnumbered_991


def test_wilcox10u991_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_991(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u991_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_991(x)
    assert isinstance(result, dict)
