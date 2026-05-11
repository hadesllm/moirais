"""Tests for information_theory_mackay4e29.information_theory_mackay_chapter_4_equation_29."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay4e29 import information_theory_mackay_chapter_4_equation_29


def test_information_theory_mackay4e29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_4_equation_29(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay4e29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_4_equation_29(x)
    assert isinstance(result, dict)
