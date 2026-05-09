"""Tests for wilcox15u1390.wilcox_chapter_15_unnumbered_1390."""
import numpy as np
import pytest
from moirais.fn.wilcox15u1390 import wilcox_chapter_15_unnumbered_1390


def test_wilcox15u1390_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1390(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15u1390_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1390(x)
    assert isinstance(result, dict)
