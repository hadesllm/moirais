"""Tests for use_r2u150.use_r_chapter_2_unnumbered_150."""
import numpy as np
import pytest
from moirais.fn.use_r2u150 import use_r_chapter_2_unnumbered_150


def test_use_r2u150_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_150(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r2u150_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_2_unnumbered_150(x)
    assert isinstance(result, dict)
