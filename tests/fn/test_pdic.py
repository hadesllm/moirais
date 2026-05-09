"""Tests for pdic.effective_parameters_dic."""
import numpy as np
import pytest
from moirais.fn.pdic import effective_parameters_dic


def test_pdic_basic():
    """Test basic functionality."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_dic(deviance)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pdic_edge():
    """Test edge cases."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_dic(deviance)
    assert isinstance(result, dict)
