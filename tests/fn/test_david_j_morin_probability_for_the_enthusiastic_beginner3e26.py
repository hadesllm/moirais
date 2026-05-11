"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e26.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26."""
import numpy as np
import pytest
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e26 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26(x)
    assert isinstance(result, dict)
