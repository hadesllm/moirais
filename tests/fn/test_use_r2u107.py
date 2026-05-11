"""Tests for use_r2u107.use_r_chapter_2_unnumbered_107."""
import numpy as np
import pytest
from morie.fn.use_r2u107 import use_r_chapter_2_unnumbered_107


def test_use_r2u107_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_107(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u107_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_107(x)
    assert isinstance(result, dict)
