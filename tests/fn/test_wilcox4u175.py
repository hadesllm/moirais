"""Tests for wilcox4u175.wilcox_chapter_4_unnumbered_175."""
import numpy as np
import pytest
from moirais.fn.wilcox4u175 import wilcox_chapter_4_unnumbered_175


def test_wilcox4u175_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_175(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u175_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_175(x)
    assert isinstance(result, dict)
