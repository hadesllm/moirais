# SPDX-License-Identifier: AGPL-3.0-or-later
"""JAX spatial GAN for synthetic crime-location generation.

A small generative-adversarial network that learns the spatial
distribution of historical crime incidents and samples synthetic
patrol/crime locations from it — the generative core of the
arXiv:2603.18987 simulation framework, reimplemented in **JAX**
(Apache-2.0) rather than PyTorch so morie stays lean and runs CPU-first
(jaxlib is ~85 MB; PyTorch's default wheels are multi-gigabyte).

The network is deliberately small (two-hidden-layer MLPs) and the
training data is standardised before fitting, which keeps adversarial
training stable enough to be deterministically testable.

This module needs the optional ``morie[sim]`` extra (``jax``).
Importing it without JAX raises :class:`ImportError`; morie's lazy
loader then reports the symbol as absent — the standard
optional-dependency behaviour.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - only hit without JAX
    raise ImportError(
        "morie.fairness.gan needs JAX — install the simulation extra: "
        "pip install 'morie[sim]'"
    ) from exc

__all__ = ["SpatialGAN"]


# ── tiny MLP ─────────────────────────────────────────────────────────

def _init_mlp(key, sizes):
    """He-initialised MLP parameters as a list of ``(W, b)`` tuples."""
    params = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (sizes[i], sizes[i + 1]))
        w = w * jnp.sqrt(2.0 / sizes[i])
        params.append((w, jnp.zeros(sizes[i + 1])))
    return params


def _mlp(params, x):
    """Forward pass; leaky-ReLU on hidden layers, linear output."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.leaky_relu(x, 0.2)
    return x


# ── hand-written Adam (keeps the extra to just `jax`) ────────────────

def _adam_init(params):
    return [(jnp.zeros_like(w), jnp.zeros_like(b),
             jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]


def _adam_step(params, grads, state, step, lr,
               b1=0.9, b2=0.999, eps=1e-8):
    new_params, new_state = [], []
    for (w, b), (gw, gb), (mw, mb, vw, vb) in zip(params, grads, state):
        mw = b1 * mw + (1 - b1) * gw
        mb = b1 * mb + (1 - b1) * gb
        vw = b2 * vw + (1 - b2) * gw ** 2
        vb = b2 * vb + (1 - b2) * gb ** 2
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        w = w - lr * (mw / bc1) / (jnp.sqrt(vw / bc2) + eps)
        b = b - lr * (mb / bc1) / (jnp.sqrt(vb / bc2) + eps)
        new_params.append((w, b))
        new_state.append((mw, mb, vw, vb))
    return new_params, new_state


# ── GAN losses ───────────────────────────────────────────────────────

def _disc_loss(dp, gp, real, z):
    fake = _mlp(gp, z)
    real_logit = _mlp(dp, real)[:, 0]
    fake_logit = _mlp(dp, fake)[:, 0]
    # binary cross-entropy: real -> 1, fake -> 0.
    # log(1 - sigmoid(x)) == log_sigmoid(-x)
    return (-jax.nn.log_sigmoid(real_logit).mean()
            - jax.nn.log_sigmoid(-fake_logit).mean())


def _gen_loss(gp, dp, z):
    fake = _mlp(gp, z)
    fake_logit = _mlp(dp, fake)[:, 0]
    # non-saturating generator loss
    return -jax.nn.log_sigmoid(fake_logit).mean()


class SpatialGAN:
    """A small JAX GAN over 2-D crime/patrol coordinates.

    Parameters
    ----------
    latent_dim : int
        Dimension of the generator's noise input.
    hidden : int
        Width of the hidden layers.
    seed : int
        Seed for parameter initialisation.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fairness.gan import SpatialGAN
    >>> rng = np.random.default_rng(0)
    >>> pts = rng.normal([5.0, -3.0], 1.0, size=(800, 2))
    >>> gan = SpatialGAN(seed=0).fit(pts, steps=400)
    >>> samples = gan.sample(500, seed=1)
    >>> samples.shape
    (500, 2)
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64,
                 seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self._gp = None          # generator params
        self._mean = None        # standardisation
        self._std = None
        self.history: list[float] = []

    def fit(self, points, *, steps: int = 1500, batch_size: int = 128,
            lr: float = 2e-3):
        """Train the GAN on an ``(n, 2)`` array of coordinates."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be an (n, 2) array")
        if pts.shape[0] < 2:
            raise ValueError("need at least two points to fit")

        self._mean = pts.mean(axis=0)
        self._std = pts.std(axis=0) + 1e-8
        std_pts = jnp.asarray((pts - self._mean) / self._std)
        n = std_pts.shape[0]

        key = jax.random.PRNGKey(self.seed)
        key, kg, kd = jax.random.split(key, 3)
        gp = _init_mlp(kg, [self.latent_dim, self.hidden, self.hidden, 2])
        dp = _init_mlp(kd, [2, self.hidden, self.hidden, 1])
        gs = _adam_init(gp)
        ds = _adam_init(dp)

        @jax.jit
        def step(gp, dp, gs, ds, t, real, zd, zg):
            dl, dg = jax.value_and_grad(_disc_loss)(dp, gp, real, zd)
            dp2, ds2 = _adam_step(dp, dg, ds, t, lr)
            gl, gg = jax.value_and_grad(_gen_loss)(gp, dp2, zg)
            gp2, gs2 = _adam_step(gp, gg, gs, t, lr)
            return gp2, dp2, gs2, ds2, dl + gl

        bs = min(batch_size, n)
        self.history = []
        for t in range(1, int(steps) + 1):
            key, ks, kzd, kzg = jax.random.split(key, 4)
            idx = jax.random.randint(ks, (bs,), 0, n)
            real = std_pts[idx]
            zd = jax.random.normal(kzd, (bs, self.latent_dim))
            zg = jax.random.normal(kzg, (bs, self.latent_dim))
            gp, dp, gs, ds, loss = step(gp, dp, gs, ds, t, real, zd, zg)
            if t % 50 == 0:
                self.history.append(float(loss))

        self._gp = gp
        return self

    def sample(self, n: int, *, seed: int | None = None):
        """Draw ``n`` synthetic coordinates as an ``(n, 2)`` numpy array."""
        if self._gp is None:
            raise RuntimeError("SpatialGAN is not fitted; call fit() first")
        key = jax.random.PRNGKey(self.seed if seed is None else int(seed))
        z = jax.random.normal(key, (int(n), self.latent_dim))
        out = np.asarray(_mlp(self._gp, z))
        return out * self._std + self._mean
