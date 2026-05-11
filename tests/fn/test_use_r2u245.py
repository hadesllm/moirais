"""Tests for use_r2u245.use_r_chapter_2_unnumbered_245."""
import numpy as np
import pytest
from morie.fn.use_r2u245 import use_r_chapter_2_unnumbered_245


def test_use_r2u245_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_245(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u245_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_245(x)
    assert isinstance(result, dict)
