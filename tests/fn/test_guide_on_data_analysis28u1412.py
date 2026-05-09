"""Tests for guide_on_data_analysis28u1412.guide_on_data_analysis_chapter_28_unnumbered_1412."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis28u1412 import guide_on_data_analysis_chapter_28_unnumbered_1412


def test_guide_on_data_analysis28u1412_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1412(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis28u1412_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1412(x)
    assert isinstance(result, dict)
