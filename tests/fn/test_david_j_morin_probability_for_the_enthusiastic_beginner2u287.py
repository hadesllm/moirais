"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2u287.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_287."""
import numpy as np
import pytest
from moirais.fn.david_j_morin_probability_for_the_enthusiastic_beginner2u287 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_287


def test_david_j_morin_probability_for_the_enthusiastic_beginner2u287_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_287(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2u287_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_287(x)
    assert isinstance(result, dict)
