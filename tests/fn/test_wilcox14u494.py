"""Tests for wilcox14u494.wilcox_chapter_14_unnumbered_494."""
import numpy as np
import pytest
from moirais.fn.wilcox14u494 import wilcox_chapter_14_unnumbered_494


def test_wilcox14u494_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_494(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u494_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_494(x)
    assert isinstance(result, dict)
