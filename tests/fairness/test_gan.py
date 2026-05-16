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

from morie.fairness.gan import SpatialGAN  # noqa: E402


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
