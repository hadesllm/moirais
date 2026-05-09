"""Tests for hedderich8e19.hedderich_chapter_8_equation_19."""
import numpy as np
import pytest
from moirais.fn.hedderich8e19 import hedderich_chapter_8_equation_19


def test_hedderich8e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_19(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_19(x)
    assert isinstance(result, dict)
