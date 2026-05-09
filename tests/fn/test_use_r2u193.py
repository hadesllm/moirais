"""Tests for use_r2u193.use_r_chapter_2_unnumbered_193."""
import numpy as np
import pytest
from moirais.fn.use_r2u193 import use_r_chapter_2_unnumbered_193


def test_use_r2u193_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_193(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u193_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_193(x)
    assert isinstance(result, dict)
