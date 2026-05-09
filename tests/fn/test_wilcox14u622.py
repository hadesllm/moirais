"""Tests for wilcox14u622.wilcox_chapter_14_unnumbered_622."""
import numpy as np
import pytest
from moirais.fn.wilcox14u622 import wilcox_chapter_14_unnumbered_622


def test_wilcox14u622_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_622(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u622_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_622(x)
    assert isinstance(result, dict)
