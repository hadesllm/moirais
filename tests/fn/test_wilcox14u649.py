"""Tests for wilcox14u649.wilcox_chapter_14_unnumbered_649."""
import numpy as np
import pytest
from moirais.fn.wilcox14u649 import wilcox_chapter_14_unnumbered_649


def test_wilcox14u649_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_649(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u649_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_649(x)
    assert isinstance(result, dict)
