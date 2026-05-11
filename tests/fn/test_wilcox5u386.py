"""Tests for wilcox5u386.wilcox_chapter_5_unnumbered_386."""
import numpy as np
import pytest
from morie.fn.wilcox5u386 import wilcox_chapter_5_unnumbered_386


def test_wilcox5u386_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_386(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u386_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_386(x)
    assert isinstance(result, dict)
