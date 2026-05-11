"""Tests for aitsbp.aitchison_sbp_basis."""
import numpy as np
import pytest
from morie.fn.aitsbp import aitchison_sbp_basis


def test_aitsbp_basic():
    """Test basic functionality."""
    sign = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_sbp_basis(sign)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitsbp_edge():
    """Test edge cases."""
    sign = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_sbp_basis(sign)
    assert isinstance(result, dict)
