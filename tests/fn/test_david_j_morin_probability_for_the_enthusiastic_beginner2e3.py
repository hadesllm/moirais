"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e3.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3."""
import numpy as np
import pytest
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e3 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3(x)
    assert isinstance(result, dict)
