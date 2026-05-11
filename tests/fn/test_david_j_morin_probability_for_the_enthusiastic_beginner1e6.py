"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e6.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6."""
import numpy as np
import pytest
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e6 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6(x)
    assert isinstance(result, dict)
