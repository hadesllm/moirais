"""Tests for wilcox14u560.wilcox_chapter_14_unnumbered_560."""
import numpy as np
import pytest
from moirais.fn.wilcox14u560 import wilcox_chapter_14_unnumbered_560


def test_wilcox14u560_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_560(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u560_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_560(x)
    assert isinstance(result, dict)
