"""Tests for jason_brownlee_machine_learning_mastery_with_r5u1.jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_1."""
import numpy as np
import pytest
from moirais.fn.jason_brownlee_machine_learning_mastery_with_r5u1 import jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_1


def test_jason_brownlee_machine_learning_mastery_with_r5u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jason_brownlee_machine_learning_mastery_with_r5u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_1(x)
    assert isinstance(result, dict)
