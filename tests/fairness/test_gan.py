# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.fairness.gan — the JAX spatial GAN.

Skipped cleanly when the optional ``morie[sim]`` extra (JAX) is not
installed.  The substantive check trains the GAN on a known Gaussian
and verifies it recovers the distribution's mean — data standardisation
makes that deterministic enough for a non-flaky test.
"""
import numpy as np
import pytest

pytest.importorskip("jax", reason="morie[sim] extra (JAX) not installed")

from morie.fairness.gan import CTGANDebiaser, SpatialGAN  # noqa: E402
from morie.fairness.metrics import fairness_disparate_impact  # noqa: E402
from morie.fairness.simulation import simulate_biased_crime_data  # noqa: E402


def test_gan_recovers_distribution_mean():
    rng = np.random.default_rng(0)
    target = np.array([5.0, -3.0])
    pts = rng.normal(target, 1.5, size=(1000, 2))
    gan = SpatialGAN(seed=0).fit(pts, steps=1200)
    samples = gan.sample(2000, seed=1)
    err = np.abs(samples.mean(axis=0) - target).max()
    assert err < 0.8, f"GAN did not recover the mean (err={err:.3f})"


def test_gan_sample_shape():
    rng = np.random.default_rng(1)
    pts = rng.normal(0.0, 1.0, size=(400, 2))
    gan = SpatialGAN(seed=0).fit(pts, steps=300)
    assert gan.sample(137, seed=2).shape == (137, 2)


def test_gan_sample_is_seeded():
    rng = np.random.default_rng(2)
    pts = rng.normal(0.0, 1.0, size=(400, 2))
    gan = SpatialGAN(seed=0).fit(pts, steps=300)
    a = gan.sample(50, seed=9)
    b = gan.sample(50, seed=9)
    assert np.array_equal(a, b)


def test_gan_sample_before_fit_raises():
    with pytest.raises(RuntimeError):
        SpatialGAN().sample(10)


def test_gan_bad_input_shape_raises():
    with pytest.raises(ValueError):
        SpatialGAN().fit(np.zeros((10, 3)))


# ── CTGANDebiaser ───────────────────────────────────────────────────

def test_ctgan_debias_reduces_disparity():
    # biased data: disparate-impact ratio ~0.4
    df = simulate_biased_crime_data(n=4000, bias=0.6, base_rate=0.4, seed=3)
    di0 = float(fairness_disparate_impact(
        df["detected"], df["group"], privileged="A"))
    assert di0 < 0.6, f"fixture should be biased (DIR={di0:.3f})"

    deb = CTGANDebiaser(seed=0).fit(
        df, outcome_col="detected", feature_cols=["risk_score"],
        group_col="group", steps=800)
    syn = deb.debias(4000, privileged="A", seed=1)
    di1 = float(fairness_disparate_impact(
        syn["detected"], syn["group"], privileged="A"))
    # rebalanced conditioning moves the DIR toward parity
    assert di1 > 0.85, f"debiasing did not reduce disparity (DIR={di1:.3f})"


def test_ctgan_debias_columns_and_size():
    df = simulate_biased_crime_data(n=800, bias=0.4, seed=5)
    deb = CTGANDebiaser(seed=0).fit(
        df, outcome_col="detected", feature_cols=["risk_score"],
        group_col="group", steps=200)
    syn = deb.debias(123, privileged="A", seed=2)
    assert len(syn) == 123
    assert set(syn.columns) == {"group", "detected", "risk_score"}


def test_ctgan_debias_before_fit_raises():
    with pytest.raises(RuntimeError):
        CTGANDebiaser().debias(10, privileged="A")


def test_ctgan_unknown_privileged_raises():
    df = simulate_biased_crime_data(n=400, seed=6)
    deb = CTGANDebiaser(seed=0).fit(
        df, outcome_col="detected", feature_cols=["risk_score"],
        group_col="group", steps=100)
    with pytest.raises(ValueError):
        deb.debias(10, privileged="Atlantis")


def test_ctgan_no_feature_columns_raises():
    df = simulate_biased_crime_data(n=400, seed=7)
    with pytest.raises(ValueError):
        CTGANDebiaser().fit(df, outcome_col="detected", feature_cols=[],
                            group_col="group")
