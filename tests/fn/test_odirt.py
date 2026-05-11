"""Test odirt."""
import numpy as np
from morie.fn.odirt import ordinal_irt_estimate


def test_odirt_basic():
    r = ordinal_irt_estimate()
    assert r.value is not None


def test_odirt_name():
    r = ordinal_irt_estimate()
    assert r.name
