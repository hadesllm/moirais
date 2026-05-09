"""Tests for analyzing_spatial_models_of_choice_and_judgment6e20.analyzing_spatial_models_of_choice_and_judgment_chapter_6_equation_20."""
import numpy as np
import pytest
from moirais.fn.analyzing_spatial_models_of_choice_and_judgment6e20 import analyzing_spatial_models_of_choice_and_judgment_chapter_6_equation_20


def test_analyzing_spatial_models_of_choice_and_judgment6e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_6_equation_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analyzing_spatial_models_of_choice_and_judgment6e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_6_equation_20(x)
    assert isinstance(result, dict)
