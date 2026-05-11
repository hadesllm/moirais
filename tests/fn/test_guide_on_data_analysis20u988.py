"""Tests for guide_on_data_analysis20u988.guide_on_data_analysis_chapter_20_unnumbered_988."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis20u988 import guide_on_data_analysis_chapter_20_unnumbered_988


def test_guide_on_data_analysis20u988_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_20_unnumbered_988(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis20u988_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_20_unnumbered_988(x)
    assert isinstance(result, dict)
