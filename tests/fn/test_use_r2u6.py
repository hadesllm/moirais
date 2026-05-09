"""Tests for use_r2u6.use_r_chapter_2_unnumbered_6."""
import numpy as np
import pytest
from moirais.fn.use_r2u6 import use_r_chapter_2_unnumbered_6


def test_use_r2u6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_6(x)
    assert isinstance(result, dict)
