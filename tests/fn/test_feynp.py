"""Tests for morie.fn.feynp -- Feynman path integral."""

import numpy as np
import pytest

from morie.fn.feynp import feynp


def test_returns_dict():
    r = feynp(x_i=0, x_f=0, T=1.0, n_paths=100, n_slices=10)
    assert isinstance(r, dict)
    for k in ("propagator_mc", "propagator_exact", "classical_action",
              "relative_error"):
        assert k in r


def test_exact_propagator_nonzero():
    r = feynp(x_i=0, x_f=0, T=1.0)
    assert abs(r["propagator_exact"]) > 0


def test_classical_action_symmetric():
    r1 = feynp(x_i=1.0, x_f=0.0, T=1.0, n_paths=10)
    r2 = feynp(x_i=0.0, x_f=1.0, T=1.0, n_paths=10)
    assert r1["classical_action"] == pytest.approx(r2["classical_action"], rel=1e-6)


def test_negative_T_raises():
    with pytest.raises(ValueError):
        feynp(x_i=0, x_f=0, T=-1.0)


def test_seed_reproducibility():
    r1 = feynp(x_i=0, x_f=1, T=1.0, n_paths=500, seed=99)
    r2 = feynp(x_i=0, x_f=1, T=1.0, n_paths=500, seed=99)
    assert r1["propagator_mc"] == r2["propagator_mc"]
