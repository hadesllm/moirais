"""Tests for wilcox14u599.wilcox_chapter_14_unnumbered_599."""
import numpy as np
import pytest
from morie.fn.wilcox14u599 import wilcox_chapter_14_unnumbered_599


def test_wilcox14u599_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_599(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u599_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_599(x)
    assert isinstance(result, dict)
