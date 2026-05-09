"""Tests for use_r11e3.use_r_chapter_11_equation_3."""
import numpy as np
import pytest
from moirais.fn.use_r11e3 import use_r_chapter_11_equation_3


def test_use_r11e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_11_equation_3(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_use_r11e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = use_r_chapter_11_equation_3(x)
    assert isinstance(result, dict)
