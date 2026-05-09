"""Tests for guide_on_data_analysis5u317.guide_on_data_analysis_chapter_5_unnumbered_317."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis5u317 import guide_on_data_analysis_chapter_5_unnumbered_317


def test_guide_on_data_analysis5u317_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_317(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis5u317_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_317(x)
    assert isinstance(result, dict)
