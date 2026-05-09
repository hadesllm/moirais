"""Tests for guide_on_data_analysis28u1405.guide_on_data_analysis_chapter_28_unnumbered_1405."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis28u1405 import guide_on_data_analysis_chapter_28_unnumbered_1405


def test_guide_on_data_analysis28u1405_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1405(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis28u1405_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1405(x)
    assert isinstance(result, dict)
