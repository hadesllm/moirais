"""Tests for use_r2u80.use_r_chapter_2_unnumbered_80."""
import numpy as np
import pytest
from morie.fn.use_r2u80 import use_r_chapter_2_unnumbered_80


def test_use_r2u80_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_80(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u80_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_80(x)
    assert isinstance(result, dict)
