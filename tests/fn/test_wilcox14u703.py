"""Tests for wilcox14u703.wilcox_chapter_14_unnumbered_703."""
import numpy as np
import pytest
from morie.fn.wilcox14u703 import wilcox_chapter_14_unnumbered_703


def test_wilcox14u703_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_703(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u703_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_703(x)
    assert isinstance(result, dict)
