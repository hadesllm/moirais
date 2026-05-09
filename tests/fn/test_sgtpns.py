"""Tests for sgtpns.sgt_perron_frobenius."""
import numpy as np
import pytest
from moirais.fn.sgtpns import sgt_perron_frobenius


def test_sgtpns_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_perron_frobenius(M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtpns_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_perron_frobenius(M)
    assert isinstance(result, dict)
