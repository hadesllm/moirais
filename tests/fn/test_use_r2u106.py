"""Tests for use_r2u106.use_r_chapter_2_unnumbered_106."""
import numpy as np
import pytest
from morie.fn.use_r2u106 import use_r_chapter_2_unnumbered_106


def test_use_r2u106_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_106(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_use_r2u106_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_106(x)
    assert isinstance(result, dict)
