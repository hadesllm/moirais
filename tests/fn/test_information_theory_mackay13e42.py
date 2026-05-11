"""Tests for information_theory_mackay13e42.information_theory_mackay_chapter_13_equation_42."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay13e42 import information_theory_mackay_chapter_13_equation_42


def test_information_theory_mackay13e42_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_13_equation_42(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay13e42_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_13_equation_42(x)
    assert isinstance(result, dict)
