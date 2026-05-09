"""Tests for guide_on_data_analysis7u600.guide_on_data_analysis_chapter_7_unnumbered_600."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis7u600 import guide_on_data_analysis_chapter_7_unnumbered_600


def test_guide_on_data_analysis7u600_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_600(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis7u600_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_600(x)
    assert isinstance(result, dict)
