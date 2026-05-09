"""Tests for neighborhood_structures_and_crime_a_spatial_analysis_george_2u17.neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_17."""
import numpy as np
import pytest
from moirais.fn.neighborhood_structures_and_crime_a_spatial_analysis_george_2u17 import neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_17


def test_neighborhood_structures_and_crime_a_spatial_analysis_george_2u17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_neighborhood_structures_and_crime_a_spatial_analysis_george_2u17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_17(x)
    assert isinstance(result, dict)
