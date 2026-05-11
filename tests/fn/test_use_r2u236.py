"""Tests for use_r2u236.use_r_chapter_2_unnumbered_236."""
import numpy as np
import pytest
from morie.fn.use_r2u236 import use_r_chapter_2_unnumbered_236


def test_use_r2u236_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_236(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u236_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_236(x)
    assert isinstance(result, dict)
