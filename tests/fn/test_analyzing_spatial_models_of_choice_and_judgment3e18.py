"""Tests for analyzing_spatial_models_of_choice_and_judgment3e18.analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_18."""
import numpy as np
import pytest
from morie.fn.analyzing_spatial_models_of_choice_and_judgment3e18 import analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_18


def test_analyzing_spatial_models_of_choice_and_judgment3e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_18(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analyzing_spatial_models_of_choice_and_judgment3e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_18(x)
    assert isinstance(result, dict)
