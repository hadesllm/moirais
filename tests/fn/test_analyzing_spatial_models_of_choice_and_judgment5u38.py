"""Tests for analyzing_spatial_models_of_choice_and_judgment5u38.analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_38."""
import numpy as np
import pytest
from morie.fn.analyzing_spatial_models_of_choice_and_judgment5u38 import analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_38


def test_analyzing_spatial_models_of_choice_and_judgment5u38_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_38(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analyzing_spatial_models_of_choice_and_judgment5u38_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_38(x)
    assert isinstance(result, dict)
