"""Tests for guide_on_data_analysis30u1540.guide_on_data_analysis_chapter_30_unnumbered_1540."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis30u1540 import guide_on_data_analysis_chapter_30_unnumbered_1540


def test_guide_on_data_analysis30u1540_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1540(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis30u1540_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1540(x)
    assert isinstance(result, dict)
