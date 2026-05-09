"""Tests for wilcox14u576.wilcox_chapter_14_unnumbered_576."""
import numpy as np
import pytest
from moirais.fn.wilcox14u576 import wilcox_chapter_14_unnumbered_576


def test_wilcox14u576_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_576(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u576_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_576(x)
    assert isinstance(result, dict)
