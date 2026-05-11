"""Test entan."""
import pytest
import numpy as np
from morie.fn.entan import entanglement_entropy


def test_entan_maximally_entangled():
    r = entanglement_entropy(schmidt_coeffs=[0.5, 0.5])
    assert r.value == pytest.approx(np.log(2))
    assert r.extra["is_maximally_entangled"]


def test_entan_pure():
    r = entanglement_entropy(schmidt_coeffs=[1.0])
    assert r.value == pytest.approx(0.0)
    assert r.extra["is_pure"]
