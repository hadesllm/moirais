"""Tests for use_r2u136.use_r_chapter_2_unnumbered_136."""
import numpy as np
import pytest
from morie.fn.use_r2u136 import use_r_chapter_2_unnumbered_136


def test_use_r2u136_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_136(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u136_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_136(x)
    assert isinstance(result, dict)
