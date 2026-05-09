"""Tests for guide_on_data_analysis25u1275.guide_on_data_analysis_chapter_25_unnumbered_1275."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis25u1275 import guide_on_data_analysis_chapter_25_unnumbered_1275


def test_guide_on_data_analysis25u1275_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1275(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis25u1275_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1275(x)
    assert isinstance(result, dict)
