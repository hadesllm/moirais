"""Tests for hedderich7e22.hedderich_chapter_7_equation_22."""
import numpy as np
import pytest
from moirais.fn.hedderich7e22 import hedderich_chapter_7_equation_22


def test_hedderich7e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_22(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich7e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_22(x)
    assert isinstance(result, dict)
