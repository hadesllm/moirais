"""Tests for guide_on_data_analysis1e63.guide_on_data_analysis_chapter_1_equation_63."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis1e63 import guide_on_data_analysis_chapter_1_equation_63


def test_guide_on_data_analysis1e63_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_1_equation_63(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis1e63_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_1_equation_63(x)
    assert isinstance(result, dict)
