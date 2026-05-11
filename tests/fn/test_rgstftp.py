"""Tests for rgstftp.rangayyan_stft_params."""
import numpy as np
import pytest
from morie.fn.rgstftp import rangayyan_stft_params


def test_rgstftp_basic():
    """Test basic functionality."""
    fs = 100.0
    desired_t_res = np.random.default_rng(42).normal(0, 1, 100)
    desired_f_res = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_stft_params(fs, desired_t_res, desired_f_res)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgstftp_edge():
    """Test edge cases."""
    fs = 100.0
    desired_t_res = np.random.default_rng(42).normal(0, 1, 100)
    desired_f_res = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_stft_params(fs, desired_t_res, desired_f_res)
    assert isinstance(result, dict)
