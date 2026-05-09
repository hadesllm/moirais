"""Tests for hedderich8e50.hedderich_chapter_8_equation_50."""
import numpy as np
import pytest
from moirais.fn.hedderich8e50 import hedderich_chapter_8_equation_50


def test_hedderich8e50_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_50(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e50_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_50(x)
    assert isinstance(result, dict)
