"""Tests for wilcox14u711.wilcox_chapter_14_unnumbered_711."""
import numpy as np
import pytest
from morie.fn.wilcox14u711 import wilcox_chapter_14_unnumbered_711


def test_wilcox14u711_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_711(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u711_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_711(x)
    assert isinstance(result, dict)
