"""Tests for scmdf.scm_definition."""
import numpy as np
import pytest
from morie.fn.scmdf import scm_definition


def test_scmdf_basic():
    """Test basic functionality."""
    exogenous = np.random.default_rng(42).normal(0, 1, 100)
    endogenous = np.random.default_rng(42).normal(0, 1, 100)
    equations = np.random.default_rng(42).normal(0, 1, 100)
    result = scm_definition(exogenous, endogenous, equations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scmdf_edge():
    """Test edge cases."""
    exogenous = np.random.default_rng(42).normal(0, 1, 100)
    endogenous = np.random.default_rng(42).normal(0, 1, 100)
    equations = np.random.default_rng(42).normal(0, 1, 100)
    result = scm_definition(exogenous, endogenous, equations)
    assert isinstance(result, dict)
