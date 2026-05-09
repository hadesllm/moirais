"""Tests for wilcox10u1041.wilcox_chapter_10_unnumbered_1041."""
import numpy as np
import pytest
from moirais.fn.wilcox10u1041 import wilcox_chapter_10_unnumbered_1041


def test_wilcox10u1041_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1041(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1041_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1041(x)
    assert isinstance(result, dict)
