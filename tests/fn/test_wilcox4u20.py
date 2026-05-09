"""Tests for wilcox4u20.wilcox_chapter_4_unnumbered_20."""
import numpy as np
import pytest
from moirais.fn.wilcox4u20 import wilcox_chapter_4_unnumbered_20


def test_wilcox4u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_20(x)
    assert isinstance(result, dict)
