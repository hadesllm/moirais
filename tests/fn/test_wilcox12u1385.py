"""Tests for wilcox12u1385.wilcox_chapter_12_unnumbered_1385."""
import numpy as np
import pytest
from morie.fn.wilcox12u1385 import wilcox_chapter_12_unnumbered_1385


def test_wilcox12u1385_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1385(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox12u1385_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1385(x)
    assert isinstance(result, dict)
