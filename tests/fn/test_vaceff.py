"""Tests for vaceff.vaccine_efficacy."""
import numpy as np
import pytest
from morie.fn.vaceff import vaccine_efficacy


def test_vaceff_basic():
    """Test basic functionality."""
    incidence_v = np.random.default_rng(42).normal(0, 1, 100)
    incidence_u = np.random.default_rng(42).normal(0, 1, 100)
    person_time = np.random.default_rng(42).normal(0, 1, 100)
    result = vaccine_efficacy(incidence_v, incidence_u, person_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vaceff_edge():
    """Test edge cases."""
    incidence_v = np.random.default_rng(42).normal(0, 1, 100)
    incidence_u = np.random.default_rng(42).normal(0, 1, 100)
    person_time = np.random.default_rng(42).normal(0, 1, 100)
    result = vaccine_efficacy(incidence_v, incidence_u, person_time)
    assert isinstance(result, dict)
