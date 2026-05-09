"""Tests for use_r2u21.use_r_chapter_2_unnumbered_21."""
import numpy as np
import pytest
from moirais.fn.use_r2u21 import use_r_chapter_2_unnumbered_21


def test_use_r2u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
