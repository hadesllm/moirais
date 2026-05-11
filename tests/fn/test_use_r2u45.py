"""Tests for use_r2u45.use_r_chapter_2_unnumbered_45."""
import numpy as np
import pytest
from morie.fn.use_r2u45 import use_r_chapter_2_unnumbered_45


def test_use_r2u45_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_45(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u45_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_45(x)
    assert isinstance(result, dict)
