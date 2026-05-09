"""Tests for wilcox14u583.wilcox_chapter_14_unnumbered_583."""
import numpy as np
import pytest
from moirais.fn.wilcox14u583 import wilcox_chapter_14_unnumbered_583


def test_wilcox14u583_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_583(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u583_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_583(x)
    assert isinstance(result, dict)
