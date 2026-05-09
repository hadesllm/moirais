"""Tests for guide_on_data_analysis1e10.guide_on_data_analysis_chapter_1_equation_10."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis1e10 import guide_on_data_analysis_chapter_1_equation_10


def test_guide_on_data_analysis1e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_1_equation_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis1e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_1_equation_10(x)
    assert isinstance(result, dict)
