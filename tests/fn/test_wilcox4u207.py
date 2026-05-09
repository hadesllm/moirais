"""Tests for wilcox4u207.wilcox_chapter_4_unnumbered_207."""
import numpy as np
import pytest
from moirais.fn.wilcox4u207 import wilcox_chapter_4_unnumbered_207


def test_wilcox4u207_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_207(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u207_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_207(x)
    assert isinstance(result, dict)
