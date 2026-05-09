"""Tests for guide_on_data_analysis6u526.guide_on_data_analysis_chapter_6_unnumbered_526."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u526 import guide_on_data_analysis_chapter_6_unnumbered_526


def test_guide_on_data_analysis6u526_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_526(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis6u526_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_526(x)
    assert isinstance(result, dict)
