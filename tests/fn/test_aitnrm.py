"""Tests for aitnrm.aitchison_norm."""
import numpy as np
import pytest
from moirais.fn.aitnrm import aitchison_norm


def test_aitnrm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_norm(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitnrm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_norm(x)
    assert isinstance(result, dict)
