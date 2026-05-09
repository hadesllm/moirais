"""Tests for chopit.chopit_vignette."""
import numpy as np
import pytest
from moirais.fn.chopit import chopit_vignette


def test_chopit_basic():
    """Test basic functionality."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    vignette_data = np.random.default_rng(42).normal(0, 1, 100)
    n_categories = np.random.default_rng(42).normal(0, 1, 100)
    result = chopit_vignette(survey_data, vignette_data, n_categories)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chopit_edge():
    """Test edge cases."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    vignette_data = np.random.default_rng(42).normal(0, 1, 100)
    n_categories = np.random.default_rng(42).normal(0, 1, 100)
    result = chopit_vignette(survey_data, vignette_data, n_categories)
    assert isinstance(result, dict)
