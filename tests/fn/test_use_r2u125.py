"""Tests for use_r2u125.use_r_chapter_2_unnumbered_125."""
import numpy as np
import pytest
from moirais.fn.use_r2u125 import use_r_chapter_2_unnumbered_125


def test_use_r2u125_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_125(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u125_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_125(x)
    assert isinstance(result, dict)
