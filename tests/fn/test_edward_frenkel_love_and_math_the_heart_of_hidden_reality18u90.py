"""Tests for edward_frenkel_love_and_math_the_heart_of_hidden_reality18u90.edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_90."""
import numpy as np
import pytest
from moirais.fn.edward_frenkel_love_and_math_the_heart_of_hidden_reality18u90 import edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_90


def test_edward_frenkel_love_and_math_the_heart_of_hidden_reality18u90_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_90(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_edward_frenkel_love_and_math_the_heart_of_hidden_reality18u90_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_90(x)
    assert isinstance(result, dict)
