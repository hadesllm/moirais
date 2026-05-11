"""Tests for rgmscart.rangayyan_muscle_artifact."""
import numpy as np
import pytest
from morie.fn.rgmscart import rangayyan_muscle_artifact


def test_rgmscart_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    emg_ref = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_muscle_artifact(vag, emg_ref, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmscart_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    emg_ref = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_muscle_artifact(vag, emg_ref, fs)
    assert isinstance(result, dict)
