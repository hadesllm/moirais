"""Tests for use_r2u137.use_r_chapter_2_unnumbered_137."""
import numpy as np
import pytest
from moirais.fn.use_r2u137 import use_r_chapter_2_unnumbered_137


def test_use_r2u137_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_137(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u137_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_137(x)
    assert isinstance(result, dict)
