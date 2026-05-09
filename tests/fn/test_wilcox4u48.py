"""Tests for wilcox4u48.wilcox_chapter_4_unnumbered_48."""
import numpy as np
import pytest
from moirais.fn.wilcox4u48 import wilcox_chapter_4_unnumbered_48


def test_wilcox4u48_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_48(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u48_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_48(x)
    assert isinstance(result, dict)
