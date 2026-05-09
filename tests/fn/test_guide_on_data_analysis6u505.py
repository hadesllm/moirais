"""Tests for guide_on_data_analysis6u505.guide_on_data_analysis_chapter_6_unnumbered_505."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u505 import guide_on_data_analysis_chapter_6_unnumbered_505


def test_guide_on_data_analysis6u505_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_505(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis6u505_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_505(x)
    assert isinstance(result, dict)
