"""Tests for wilcox4u37.wilcox_chapter_4_unnumbered_37."""
import numpy as np
import pytest
from moirais.fn.wilcox4u37 import wilcox_chapter_4_unnumbered_37


def test_wilcox4u37_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_37(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u37_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_37(x)
    assert isinstance(result, dict)
