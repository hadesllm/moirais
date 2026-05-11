"""Test vnamp."""
import pytest
from morie.fn.vnamp import veneziano_amplitude


def test_vnamp_basic():
    r = veneziano_amplitude(s=1.0, t=-0.5)
    assert r.value is not None
    assert r.value > 0


def test_vnamp_name():
    r = veneziano_amplitude()
    assert r.name == "veneziano_amplitude"
