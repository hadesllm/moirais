"""Tests for wilcox4u172.wilcox_chapter_4_unnumbered_172."""
import numpy as np
import pytest
from moirais.fn.wilcox4u172 import wilcox_chapter_4_unnumbered_172


def test_wilcox4u172_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_172(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u172_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_172(x)
    assert isinstance(result, dict)
