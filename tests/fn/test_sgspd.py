"""Test sgspd."""
import numpy as np
from moirais.fn.sgspd import space_deformation


def test_sgspd_basic():
    r = space_deformation()
    assert r.statistic is not None


def test_sgspd_name():
    r = space_deformation()
    assert r.name
