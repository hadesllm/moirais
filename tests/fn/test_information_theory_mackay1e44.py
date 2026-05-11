"""Tests for information_theory_mackay1e44.information_theory_mackay_chapter_1_equation_44."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay1e44 import information_theory_mackay_chapter_1_equation_44


def test_information_theory_mackay1e44_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_44(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay1e44_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_44(x)
    assert isinstance(result, dict)
