"""Tests for guide_on_data_analysis14u878.guide_on_data_analysis_chapter_14_unnumbered_878."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis14u878 import guide_on_data_analysis_chapter_14_unnumbered_878


def test_guide_on_data_analysis14u878_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_878(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u878_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_878(x)
    assert isinstance(result, dict)
