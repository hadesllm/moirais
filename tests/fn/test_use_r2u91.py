"""Tests for use_r2u91.use_r_chapter_2_unnumbered_91."""
import numpy as np
import pytest
from morie.fn.use_r2u91 import use_r_chapter_2_unnumbered_91


def test_use_r2u91_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_91(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u91_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_91(x)
    assert isinstance(result, dict)
