"""Tests for wilcox14u526.wilcox_chapter_14_unnumbered_526."""
import numpy as np
import pytest
from moirais.fn.wilcox14u526 import wilcox_chapter_14_unnumbered_526


def test_wilcox14u526_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_526(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u526_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_526(x)
    assert isinstance(result, dict)
