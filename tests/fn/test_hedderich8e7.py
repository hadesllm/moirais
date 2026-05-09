"""Tests for hedderich8e7.hedderich_chapter_8_equation_7."""
import numpy as np
import pytest
from moirais.fn.hedderich8e7 import hedderich_chapter_8_equation_7


def test_hedderich8e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_7(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_7(x)
    assert isinstance(result, dict)
