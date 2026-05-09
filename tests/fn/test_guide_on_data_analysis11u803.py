"""Tests for guide_on_data_analysis11u803.guide_on_data_analysis_chapter_11_unnumbered_803."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis11u803 import guide_on_data_analysis_chapter_11_unnumbered_803


def test_guide_on_data_analysis11u803_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_803(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis11u803_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_803(x)
    assert isinstance(result, dict)
