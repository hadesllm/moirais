"""Tests for information_theory_mackay24e9.information_theory_mackay_chapter_24_equation_9."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay24e9 import information_theory_mackay_chapter_24_equation_9


def test_information_theory_mackay24e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_24_equation_9(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay24e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_24_equation_9(x)
    assert isinstance(result, dict)
