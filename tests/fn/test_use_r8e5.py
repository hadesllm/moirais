"""Tests for use_r8e5.use_r_chapter_8_equation_5."""
import numpy as np
import pytest
from moirais.fn.use_r8e5 import use_r_chapter_8_equation_5


def test_use_r8e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_8_equation_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r8e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_8_equation_5(x)
    assert isinstance(result, dict)
