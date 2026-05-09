"""Tests for moirais.fn.lokiv -- GAN discriminator score."""

import numpy as np
from moirais.fn.lokiv import illusion_score, lokiv
from moirais.fn._containers import DescriptiveResult


class TestLokiv:
    def test_alias(self):
        assert lokiv is illusion_score

    def test_perfect_discriminator(self):
        real = np.array([0.95, 0.99, 0.92, 0.97])
        fake = np.array([0.05, 0.01, 0.08, 0.03])
        r = illusion_score(real, fake)
        assert isinstance(r, DescriptiveResult)
        assert r.value["accuracy"] > 0.9

    def test_confused_discriminator(self):
        rng = np.random.default_rng(42)
        real = rng.uniform(0.3, 0.7, 50)
        fake = rng.uniform(0.3, 0.7, 50)
        r = illusion_score(real, fake)
        assert r.value["accuracy"] < 0.7
