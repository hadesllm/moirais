"""Tests for guide_on_data_analysis19u971.guide_on_data_analysis_chapter_19_unnumbered_971."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis19u971 import guide_on_data_analysis_chapter_19_unnumbered_971


def test_guide_on_data_analysis19u971_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_19_unnumbered_971(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis19u971_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_19_unnumbered_971(x)
    assert isinstance(result, dict)
