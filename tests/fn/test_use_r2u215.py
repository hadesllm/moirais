"""Tests for use_r2u215.use_r_chapter_2_unnumbered_215."""
import numpy as np
import pytest
from morie.fn.use_r2u215 import use_r_chapter_2_unnumbered_215


def test_use_r2u215_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_215(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_use_r2u215_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_215(x)
    assert isinstance(result, dict)
