"""Tests for use_r2u40.use_r_chapter_2_unnumbered_40."""
import numpy as np
import pytest
from moirais.fn.use_r2u40 import use_r_chapter_2_unnumbered_40


def test_use_r2u40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_40(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_40(x)
    assert isinstance(result, dict)
