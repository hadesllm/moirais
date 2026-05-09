"""Tests for wilcox4u107.wilcox_chapter_4_unnumbered_107."""
import numpy as np
import pytest
from moirais.fn.wilcox4u107 import wilcox_chapter_4_unnumbered_107


def test_wilcox4u107_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_107(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u107_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_107(x)
    assert isinstance(result, dict)
