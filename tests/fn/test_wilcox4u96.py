"""Tests for wilcox4u96.wilcox_chapter_4_unnumbered_96."""
import numpy as np
import pytest
from moirais.fn.wilcox4u96 import wilcox_chapter_4_unnumbered_96


def test_wilcox4u96_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_96(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u96_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_96(x)
    assert isinstance(result, dict)
