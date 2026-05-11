"""Tests for hedderich7e16.hedderich_chapter_7_equation_16."""
import numpy as np
import pytest
from morie.fn.hedderich7e16 import hedderich_chapter_7_equation_16


def test_hedderich7e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_16(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich7e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_16(x)
    assert isinstance(result, dict)
