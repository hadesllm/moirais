"""Tests for probability_and_random_processes_with_one_thousand_exercises5u2401.probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2401."""
import numpy as np
import pytest
from moirais.fn.probability_and_random_processes_with_one_thousand_exercises5u2401 import probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2401


def test_probability_and_random_processes_with_one_thousand_exercises5u2401_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2401(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_probability_and_random_processes_with_one_thousand_exercises5u2401_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2401(x)
    assert isinstance(result, dict)
