"""Test smcof."""
import numpy as np
from morie.fn.smcof import smacof_scale


def test_smcof_basic():
    r = smacof_scale()
    assert r.value is not None


def test_smcof_name():
    r = smacof_scale()
    assert r.name
