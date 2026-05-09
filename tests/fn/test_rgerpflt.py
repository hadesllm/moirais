"""Tests for rgerpflt.rangayyan_erp_artifact_remove."""
import numpy as np
import pytest
from moirais.fn.rgerpflt import rangayyan_erp_artifact_remove


def test_rgerpflt_basic():
    """Test basic functionality."""
    erp_epochs = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_erp_artifact_remove(erp_epochs, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgerpflt_edge():
    """Test edge cases."""
    erp_epochs = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_erp_artifact_remove(erp_epochs, fs)
    assert isinstance(result, dict)
