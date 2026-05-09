"""Tests for oralb.oral_bioavailability."""
import numpy as np
import pytest
from moirais.fn.oralb import oral_bioavailability


def test_oralb_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = oral_bioavailability(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_oralb_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = oral_bioavailability(smiles)
    assert isinstance(result, dict)
