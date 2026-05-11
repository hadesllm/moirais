"""Tests for morie.fn.bdiag — MCMC diagnostics."""

import numpy as np
import pytest

from morie.fn.bdiag import mcmc_diagnostics


class TestMCMCDiagnostics:
    def test_converged_chains(self):
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (4, 5000))
        res = mcmc_diagnostics(chains)
        assert res.extra["rhat"] < 1.1

    def test_single_chain(self):
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, 2000)
        res = mcmc_diagnostics(chain.reshape(1, -1))
        assert res.extra["ess"] > 0

    def test_ess_positive(self):
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (2, 3000))
        res = mcmc_diagnostics(chains)
        assert res.extra["ess"] > 100
