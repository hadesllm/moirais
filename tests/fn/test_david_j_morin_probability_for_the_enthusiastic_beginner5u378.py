"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5u378.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_378."""
import numpy as np
import pytest
from moirais.fn.david_j_morin_probability_for_the_enthusiastic_beginner5u378 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_378


def test_david_j_morin_probability_for_the_enthusiastic_beginner5u378_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_378(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5u378_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_378(x)
    assert isinstance(result, dict)
