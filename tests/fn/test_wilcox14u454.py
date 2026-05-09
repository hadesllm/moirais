"""Tests for wilcox14u454.wilcox_chapter_14_unnumbered_454."""
import numpy as np
import pytest
from moirais.fn.wilcox14u454 import wilcox_chapter_14_unnumbered_454


def test_wilcox14u454_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_454(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u454_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_454(x)
    assert isinstance(result, dict)
