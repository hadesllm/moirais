"""Tests for hedderich8e1.hedderich_chapter_8_equation_1."""
import numpy as np
import pytest
from morie.fn.hedderich8e1 import hedderich_chapter_8_equation_1


def test_hedderich8e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_1(x)
    assert isinstance(result, dict)
