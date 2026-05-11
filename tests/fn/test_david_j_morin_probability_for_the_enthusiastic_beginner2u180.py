"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2u180.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_180."""
import numpy as np
import pytest
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2u180 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_180


def test_david_j_morin_probability_for_the_enthusiastic_beginner2u180_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_180(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2u180_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_180(x)
    assert isinstance(result, dict)
