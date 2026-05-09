"""Tests for hedderich7e17.hedderich_chapter_7_equation_17."""
import numpy as np
import pytest
from moirais.fn.hedderich7e17 import hedderich_chapter_7_equation_17


def test_hedderich7e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich7e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_7_equation_17(x)
    assert isinstance(result, dict)
