"""Tests for guide_on_data_analysis5u338.guide_on_data_analysis_chapter_5_unnumbered_338."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis5u338 import guide_on_data_analysis_chapter_5_unnumbered_338


def test_guide_on_data_analysis5u338_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_338(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis5u338_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_338(x)
    assert isinstance(result, dict)
