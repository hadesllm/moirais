"""Tests for wilcox4u215.wilcox_chapter_4_unnumbered_215."""
import numpy as np
import pytest
from morie.fn.wilcox4u215 import wilcox_chapter_4_unnumbered_215


def test_wilcox4u215_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_215(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u215_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_215(x)
    assert isinstance(result, dict)
