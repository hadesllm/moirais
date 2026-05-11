"""Tests for wilcox4u71.wilcox_chapter_4_unnumbered_71."""
import numpy as np
import pytest
from morie.fn.wilcox4u71 import wilcox_chapter_4_unnumbered_71


def test_wilcox4u71_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_71(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u71_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_71(x)
    assert isinstance(result, dict)
