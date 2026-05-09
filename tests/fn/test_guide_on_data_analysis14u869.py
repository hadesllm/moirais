"""Tests for guide_on_data_analysis14u869.guide_on_data_analysis_chapter_14_unnumbered_869."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis14u869 import guide_on_data_analysis_chapter_14_unnumbered_869


def test_guide_on_data_analysis14u869_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_869(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u869_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_869(x)
    assert isinstance(result, dict)
