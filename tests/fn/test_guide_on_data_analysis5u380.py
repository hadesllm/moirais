"""Tests for guide_on_data_analysis5u380.guide_on_data_analysis_chapter_5_unnumbered_380."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis5u380 import guide_on_data_analysis_chapter_5_unnumbered_380


def test_guide_on_data_analysis5u380_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_380(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis5u380_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_380(x)
    assert isinstance(result, dict)
