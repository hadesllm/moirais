"""Tests for prime_density."""
import pytest
from moirais.fn.prime import prime_density

class TestPrime:
    def test_known(self):
        r = prime_density(100)
        assert r.extra["pi_n"] == 25

    def test_small(self):
        r = prime_density(10)
        assert r.extra["pi_n"] == 4
