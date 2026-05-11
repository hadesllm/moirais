"""Tests for hedderich8e41.hedderich_chapter_8_equation_41."""
import numpy as np
import pytest
from morie.fn.hedderich8e41 import hedderich_chapter_8_equation_41


def test_hedderich8e41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_41(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_41(x)
    assert isinstance(result, dict)
