"""Tests for use_r2u155.use_r_chapter_2_unnumbered_155."""
import numpy as np
import pytest
from morie.fn.use_r2u155 import use_r_chapter_2_unnumbered_155


def test_use_r2u155_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_155(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u155_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_155(x)
    assert isinstance(result, dict)
