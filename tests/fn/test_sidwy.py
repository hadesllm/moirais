"""Tests for morie.fn.sidwy -- adversarial perturbation."""

import numpy as np
from morie.fn.sidwy import adversarial_perturb, sidwy
from morie.fn._containers import DescriptiveResult


class TestSidwy:
    def test_alias(self):
        assert sidwy is adversarial_perturb

    def test_fgsm(self):
        x = np.array([1.0, 2.0, 3.0])
        g = np.array([0.5, -0.3, 0.8])
        r = adversarial_perturb(x, g, epsilon=0.1)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["linf_norm"] <= 0.1 + 1e-10

    def test_pgd(self):
        x = np.zeros(10)
        g = np.ones(10)
        r = adversarial_perturb(x, g, epsilon=1.0, method="pgd_step")
        assert r.extra["l2_norm"] <= 1.0 + 1e-10
