"""Tests for moirais.fn.boltz -- Boltzmann distribution."""

import numpy as np
import pytest

from moirais.fn.boltz import boltz


def test_returns_dict():
    E = np.array([0.0, 1.0, 2.0]) * 1e-21
    r = boltz(E, T=300.0)
    assert isinstance(r, dict)
    for k in ("probabilities", "partition_function", "mean_energy",
              "entropy", "free_energy"):
        assert k in r


def test_probabilities_sum_to_one():
    E = np.array([0.0, 1.0, 2.0, 3.0]) * 1e-21
    r = boltz(E, T=300.0)
    assert np.sum(r["probabilities"]) == pytest.approx(1.0, abs=1e-10)


def test_ground_state_dominates_low_T():
    E = np.array([0.0, 1.0]) * 1e-20
    r = boltz(E, T=0.001)
    assert r["probabilities"][0] > 0.999


def test_degeneracy():
    E = np.array([0.0, 1e-21])
    r1 = boltz(E, T=300.0, degeneracies=np.array([1, 1]))
    r2 = boltz(E, T=300.0, degeneracies=np.array([1, 3]))
    assert r2["probabilities"][1] > r1["probabilities"][1]


def test_zero_temp_raises():
    with pytest.raises(ValueError):
        boltz(np.array([0.0, 1.0]), T=0.0)
