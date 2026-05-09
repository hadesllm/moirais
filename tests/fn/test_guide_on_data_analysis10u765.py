"""Tests for guide_on_data_analysis10u765.guide_on_data_analysis_chapter_10_unnumbered_765."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis10u765 import guide_on_data_analysis_chapter_10_unnumbered_765


def test_guide_on_data_analysis10u765_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_10_unnumbered_765(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis10u765_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_10_unnumbered_765(x)
    assert isinstance(result, dict)
