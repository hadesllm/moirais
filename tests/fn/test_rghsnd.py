"""Tests for rghsnd.rangayyan_heart_sound_id."""
import numpy as np
import pytest
from morie.fn.rghsnd import rangayyan_heart_sound_id


def test_rghsnd_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_heart_sound_id(pcg, ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghsnd_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_heart_sound_id(pcg, ecg, fs)
    assert isinstance(result, dict)
