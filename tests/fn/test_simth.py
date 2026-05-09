"""Tests for moirais.fn.simth -- simulate thermometer."""

import numpy as np
from moirais.fn.simth import simulate_thermometer, simth


def test_simth_smoke():
    r = simth(n_resp=20, n_stim=5)
    assert r.name == "simulate_thermometer"
    assert r.value.shape == (20, 5)
    assert np.all(r.value >= 0)
    assert np.all(r.value <= 100)


def test_simth_alias():
    assert simth is simulate_thermometer
