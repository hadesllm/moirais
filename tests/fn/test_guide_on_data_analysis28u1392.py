"""Tests for guide_on_data_analysis28u1392.guide_on_data_analysis_chapter_28_unnumbered_1392."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis28u1392 import guide_on_data_analysis_chapter_28_unnumbered_1392


def test_guide_on_data_analysis28u1392_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1392(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis28u1392_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1392(x)
    assert isinstance(result, dict)
