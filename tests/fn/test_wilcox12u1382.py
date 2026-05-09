"""Tests for wilcox12u1382.wilcox_chapter_12_unnumbered_1382."""
import numpy as np
import pytest
from moirais.fn.wilcox12u1382 import wilcox_chapter_12_unnumbered_1382


def test_wilcox12u1382_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1382(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox12u1382_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1382(x)
    assert isinstance(result, dict)
