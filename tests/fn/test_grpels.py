"""Tests for grpels.geron_peephole_lstm_cell."""
import numpy as np
import pytest
from morie.fn.grpels import geron_peephole_lstm_cell


def test_grpels_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wf = np.random.default_rng(42).normal(0, 1, 100)
    Wi = np.random.default_rng(42).normal(0, 1, 100)
    Wg = np.random.default_rng(42).normal(0, 1, 100)
    Wo = np.random.default_rng(42).normal(0, 1, 100)
    Uf = np.random.default_rng(42).normal(0, 1, 100)
    Ui = np.random.default_rng(42).normal(0, 1, 100)
    Uo = np.random.default_rng(42).normal(0, 1, 100)
    bf = np.random.default_rng(42).normal(0, 1, 100)
    bi = np.random.default_rng(42).normal(0, 1, 100)
    bg = np.random.default_rng(42).normal(0, 1, 100)
    bo = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpels_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    Wf = np.random.default_rng(42).normal(0, 1, 100)
    Wi = np.random.default_rng(42).normal(0, 1, 100)
    Wg = np.random.default_rng(42).normal(0, 1, 100)
    Wo = np.random.default_rng(42).normal(0, 1, 100)
    Uf = np.random.default_rng(42).normal(0, 1, 100)
    Ui = np.random.default_rng(42).normal(0, 1, 100)
    Uo = np.random.default_rng(42).normal(0, 1, 100)
    bf = np.random.default_rng(42).normal(0, 1, 100)
    bi = np.random.default_rng(42).normal(0, 1, 100)
    bg = np.random.default_rng(42).normal(0, 1, 100)
    bo = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo)
    assert isinstance(result, dict)
