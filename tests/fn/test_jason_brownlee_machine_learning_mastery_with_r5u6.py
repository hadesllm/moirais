"""Tests for jason_brownlee_machine_learning_mastery_with_r5u6.jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_6."""
import numpy as np
import pytest
from morie.fn.jason_brownlee_machine_learning_mastery_with_r5u6 import jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_6


def test_jason_brownlee_machine_learning_mastery_with_r5u6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jason_brownlee_machine_learning_mastery_with_r5u6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_6(x)
    assert isinstance(result, dict)
