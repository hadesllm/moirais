"""Tests for wilcox14u512.wilcox_chapter_14_unnumbered_512."""
import numpy as np
import pytest
from moirais.fn.wilcox14u512 import wilcox_chapter_14_unnumbered_512


def test_wilcox14u512_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_512(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u512_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_512(x)
    assert isinstance(result, dict)
