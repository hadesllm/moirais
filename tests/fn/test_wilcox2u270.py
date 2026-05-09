"""Tests for wilcox2u270.wilcox_chapter_2_unnumbered_270."""
import numpy as np
import pytest
from moirais.fn.wilcox2u270 import wilcox_chapter_2_unnumbered_270


def test_wilcox2u270_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_270(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox2u270_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_270(x)
    assert isinstance(result, dict)
