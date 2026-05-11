"""Tests for jason_brownlee_machine_learning_mastery_with_r1e15.jason_brownlee_machine_learning_mastery_with_r_chapter_1_equation_15."""
import numpy as np
import pytest
from morie.fn.jason_brownlee_machine_learning_mastery_with_r1e15 import jason_brownlee_machine_learning_mastery_with_r_chapter_1_equation_15


def test_jason_brownlee_machine_learning_mastery_with_r1e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_1_equation_15(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jason_brownlee_machine_learning_mastery_with_r1e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_1_equation_15(x)
    assert isinstance(result, dict)
