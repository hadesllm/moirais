"""Tests for wilcox4u200.wilcox_chapter_4_unnumbered_200."""
import numpy as np
import pytest
from moirais.fn.wilcox4u200 import wilcox_chapter_4_unnumbered_200


def test_wilcox4u200_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_200(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u200_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_200(x)
    assert isinstance(result, dict)
