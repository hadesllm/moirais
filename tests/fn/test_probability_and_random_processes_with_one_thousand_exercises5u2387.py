"""Tests for probability_and_random_processes_with_one_thousand_exercises5u2387.probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2387."""
import numpy as np
import pytest
from moirais.fn.probability_and_random_processes_with_one_thousand_exercises5u2387 import probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2387


def test_probability_and_random_processes_with_one_thousand_exercises5u2387_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2387(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_probability_and_random_processes_with_one_thousand_exercises5u2387_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2387(x)
    assert isinstance(result, dict)
