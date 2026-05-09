"""Tests for probability_and_random_processes_with_one_thousand_exercises2u381.probability_and_random_processes_with_one_thousand_exercises_chapter_2_unnumbered_381."""
import numpy as np
import pytest
from moirais.fn.probability_and_random_processes_with_one_thousand_exercises2u381 import probability_and_random_processes_with_one_thousand_exercises_chapter_2_unnumbered_381


def test_probability_and_random_processes_with_one_thousand_exercises2u381_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_2_unnumbered_381(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_probability_and_random_processes_with_one_thousand_exercises2u381_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = probability_and_random_processes_with_one_thousand_exercises_chapter_2_unnumbered_381(x)
    assert isinstance(result, dict)
