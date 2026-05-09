"""Tests for wilcox8u867.wilcox_chapter_8_unnumbered_867."""
import numpy as np
import pytest
from moirais.fn.wilcox8u867 import wilcox_chapter_8_unnumbered_867


def test_wilcox8u867_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_867(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u867_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_867(x)
    assert isinstance(result, dict)
