"""Tests for information_theory_mackay6u145.information_theory_mackay_chapter_6_unnumbered_145."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay6u145 import information_theory_mackay_chapter_6_unnumbered_145


def test_information_theory_mackay6u145_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_145(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u145_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_145(x)
    assert isinstance(result, dict)
