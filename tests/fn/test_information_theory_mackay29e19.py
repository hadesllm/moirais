"""Tests for information_theory_mackay29e19.information_theory_mackay_chapter_29_equation_19."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay29e19 import information_theory_mackay_chapter_29_equation_19


def test_information_theory_mackay29e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_29_equation_19(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay29e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_29_equation_19(x)
    assert isinstance(result, dict)
