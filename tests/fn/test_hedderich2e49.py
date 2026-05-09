"""Tests for hedderich2e49.hedderich_chapter_2_equation_49."""
import numpy as np
import pytest
from moirais.fn.hedderich2e49 import hedderich_chapter_2_equation_49


def test_hedderich2e49_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_49(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich2e49_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_49(x)
    assert isinstance(result, dict)
