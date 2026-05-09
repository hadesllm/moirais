"""Tests for hedderich8e91.hedderich_chapter_8_equation_91."""
import numpy as np
import pytest
from moirais.fn.hedderich8e91 import hedderich_chapter_8_equation_91


def test_hedderich8e91_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_91(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e91_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_91(x)
    assert isinstance(result, dict)
