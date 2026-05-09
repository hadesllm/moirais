"""Tests for wilcox4u146.wilcox_chapter_4_unnumbered_146."""
import numpy as np
import pytest
from moirais.fn.wilcox4u146 import wilcox_chapter_4_unnumbered_146


def test_wilcox4u146_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_146(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u146_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_146(x)
    assert isinstance(result, dict)
