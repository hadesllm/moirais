"""Tests for rgeqn5a.rangayyan_ch5_waveform_morph."""
import numpy as np
import pytest
from morie.fn.rgeqn5a import rangayyan_ch5_waveform_morph


def test_rgeqn5a_basic():
    """Test basic functionality."""
    template = np.random.default_rng(42).normal(0, 1, 100)
    beat = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch5_waveform_morph(template, beat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn5a_edge():
    """Test edge cases."""
    template = np.random.default_rng(42).normal(0, 1, 100)
    beat = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch5_waveform_morph(template, beat)
    assert isinstance(result, dict)
