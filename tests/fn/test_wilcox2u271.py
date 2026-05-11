"""Tests for wilcox2u271.wilcox_chapter_2_unnumbered_271."""
import numpy as np
import pytest
from morie.fn.wilcox2u271 import wilcox_chapter_2_unnumbered_271


def test_wilcox2u271_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_271(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u271_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_271(x)
    assert isinstance(result, dict)
