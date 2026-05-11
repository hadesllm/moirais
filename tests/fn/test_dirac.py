"""Tests for morie.fn.dirac -- Dirac equation."""

import numpy as np
import pytest

from morie.fn.dirac import dirac


def test_returns_dict():
    r = dirac(p=np.array([0, 0, 1e-24]), m=9.109e-31)
    assert isinstance(r, dict)
    for k in ("energy", "spinor", "gamma_matrices"):
        assert k in r


def test_rest_energy():
    m = 9.109e-31
    c = 299792458.0
    r = dirac(p=np.zeros(3), m=m)
    assert r["energy"] == pytest.approx(m * c ** 2, rel=1e-10)


def test_spinor_length_4():
    r = dirac(p=np.array([1e-24, 0, 0]), m=9.109e-31)
    assert r["spinor"].shape == (4,)


def test_gamma_matrices_count():
    r = dirac(p=np.zeros(3), m=9.109e-31)
    assert len(r["gamma_matrices"]) == 4
    for g in r["gamma_matrices"]:
        assert g.shape == (2, 2) or g.shape == (4, 4)


def test_antiparticle():
    r = dirac(p=np.array([1e-24, 0, 0]), m=9.109e-31, particle="antiparticle")
    assert r["energy"] > 0


def test_wrong_p_shape():
    with pytest.raises(ValueError):
        dirac(p=np.zeros(4), m=1.0)
