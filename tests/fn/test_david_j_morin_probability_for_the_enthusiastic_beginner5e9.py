"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e9.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_9."""
import numpy as np
import pytest
from moirais.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e9 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_9


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_9(x)
    assert isinstance(result, dict)
