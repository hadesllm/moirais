"""Tests for temperature schedule."""
import numpy as np
from morie.fn.sgtmp import sgtmp


def test_sgtmp_smoke():
    r = sgtmp(T0=1.0, cooling=0.9, n_iter=50)
    assert r.name == "temperature_schedule"
    assert len(r.extra["schedule"]) == 50
    assert r.extra["schedule"][0] == 1.0
    assert r.extra["schedule"][-1] < 0.01


def test_sgtmp_monotone():
    r = sgtmp(T0=10.0, cooling=0.95, n_iter=100)
    schedule = r.extra["schedule"]
    assert np.all(np.diff(schedule) <= 0)
