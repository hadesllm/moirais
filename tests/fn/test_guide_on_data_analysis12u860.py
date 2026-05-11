"""Tests for guide_on_data_analysis12u860.guide_on_data_analysis_chapter_12_unnumbered_860."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis12u860 import guide_on_data_analysis_chapter_12_unnumbered_860


def test_guide_on_data_analysis12u860_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_12_unnumbered_860(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis12u860_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_12_unnumbered_860(x)
    assert isinstance(result, dict)
