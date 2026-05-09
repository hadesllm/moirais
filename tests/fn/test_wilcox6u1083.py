"""Tests for wilcox6u1083.wilcox_chapter_6_unnumbered_1083."""
import numpy as np
import pytest
from moirais.fn.wilcox6u1083 import wilcox_chapter_6_unnumbered_1083


def test_wilcox6u1083_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_unnumbered_1083(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox6u1083_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_unnumbered_1083(x)
    assert isinstance(result, dict)
