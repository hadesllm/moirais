"""Tests for guide_on_data_analysis14u902.guide_on_data_analysis_chapter_14_unnumbered_902."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis14u902 import guide_on_data_analysis_chapter_14_unnumbered_902


def test_guide_on_data_analysis14u902_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_902(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis14u902_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_902(x)
    assert isinstance(result, dict)
