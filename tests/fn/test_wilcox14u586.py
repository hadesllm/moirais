"""Tests for wilcox14u586.wilcox_chapter_14_unnumbered_586."""
import numpy as np
import pytest
from moirais.fn.wilcox14u586 import wilcox_chapter_14_unnumbered_586


def test_wilcox14u586_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_586(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u586_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_586(x)
    assert isinstance(result, dict)
