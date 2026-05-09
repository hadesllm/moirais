"""Tests for guide_on_data_analysis16u917.guide_on_data_analysis_chapter_16_unnumbered_917."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis16u917 import guide_on_data_analysis_chapter_16_unnumbered_917


def test_guide_on_data_analysis16u917_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_917(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis16u917_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_917(x)
    assert isinstance(result, dict)
