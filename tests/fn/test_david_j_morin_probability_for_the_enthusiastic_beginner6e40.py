"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e40.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40."""
import numpy as np
import pytest
from moirais.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e40 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40(x)
    assert isinstance(result, dict)
