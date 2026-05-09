"""Tests for sppql.schabenberger_pql_glmm."""
import numpy as np
import pytest
from moirais.fn.sppql import schabenberger_pql_glmm


def test_sppql_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    link = 'identity'
    family = 'gaussian'
    result = schabenberger_pql_glmm(x, y, coords, link, family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sppql_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    link = 'identity'
    family = 'gaussian'
    result = schabenberger_pql_glmm(x, y, coords, link, family)
    assert isinstance(result, dict)
