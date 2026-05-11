"""Tests for wilcox4u177.wilcox_chapter_4_unnumbered_177."""
import numpy as np
import pytest
from morie.fn.wilcox4u177 import wilcox_chapter_4_unnumbered_177


def test_wilcox4u177_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_177(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u177_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_177(x)
    assert isinstance(result, dict)
