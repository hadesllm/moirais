"""Tests for morie.fn.prmsc -- VAE sampling."""

import numpy as np
from morie.fn.prmsc import vae_sample, prmsc
from morie.fn._containers import DescriptiveResult


class TestPrmsc:
    def test_alias(self):
        assert prmsc is vae_sample

    def test_shape(self):
        mu = np.zeros((5, 3))
        log_var = np.zeros((5, 3))
        r = vae_sample(mu, log_var)
        assert isinstance(r, DescriptiveResult)
        assert r.value.shape == (5, 3)

    def test_kl_zero_for_standard(self):
        mu = np.zeros((10, 4))
        log_var = np.zeros((10, 4))
        r = vae_sample(mu, log_var)
        assert abs(r.extra["kl_divergence"]) < 0.01
