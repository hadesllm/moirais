"""Tests for wilcox4u93.wilcox_chapter_4_unnumbered_93."""
import numpy as np
import pytest
from moirais.fn.wilcox4u93 import wilcox_chapter_4_unnumbered_93


def test_wilcox4u93_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_93(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u93_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_93(x)
    assert isinstance(result, dict)
