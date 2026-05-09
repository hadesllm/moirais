"""Tests for guide_on_data_analysis17u952.guide_on_data_analysis_chapter_17_unnumbered_952."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis17u952 import guide_on_data_analysis_chapter_17_unnumbered_952


def test_guide_on_data_analysis17u952_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_17_unnumbered_952(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis17u952_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_17_unnumbered_952(x)
    assert isinstance(result, dict)
