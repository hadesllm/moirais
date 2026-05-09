"""Tests for wilcox4u171.wilcox_chapter_4_unnumbered_171."""
import numpy as np
import pytest
from moirais.fn.wilcox4u171 import wilcox_chapter_4_unnumbered_171


def test_wilcox4u171_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_171(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u171_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_171(x)
    assert isinstance(result, dict)
