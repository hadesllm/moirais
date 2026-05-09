"""Tests for moirais.fn.vtbal — Viterbi alignment."""
import numpy as np
import pytest

from moirais.fn.vtbal import viterbi_align, vtbal


def test_deterministic_path():
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])
    obs = np.array([0, 0, 1, 1, 0])
    result = viterbi_align(obs, 2, trans, emit)
    assert len(result.extra["path"]) == 5
    assert result.extra["n_states"] == 2
    assert result.extra["n_observations"] == 5


def test_single_state():
    trans = np.array([[1.0]])
    emit = np.array([[0.5, 0.5]])
    obs = np.array([0, 1, 0])
    result = viterbi_align(obs, 1, trans, emit)
    assert np.all(result.extra["path"] == 0)


def test_log_probability_finite():
    rng = np.random.default_rng(42)
    trans = np.array([[0.6, 0.4], [0.3, 0.7]])
    emit = np.array([[0.8, 0.2], [0.1, 0.9]])
    obs = rng.choice(2, size=20)
    result = viterbi_align(obs, 2, trans, emit)
    assert np.isfinite(result.value)


def test_alias():
    assert vtbal is viterbi_align
