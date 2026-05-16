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

__all__ = ["SpatialGAN", "CTGANDebiaser"]


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


# ── conditional GAN losses (for the CTGAN-style debiaser) ────────────

def _cond_disc_loss(dp, gp, real_feat, cond, z):
    fake = _mlp(gp, jnp.concatenate([z, cond], axis=1))
    real_logit = _mlp(dp, jnp.concatenate([real_feat, cond], axis=1))[:, 0]
    fake_logit = _mlp(dp, jnp.concatenate([fake, cond], axis=1))[:, 0]
    return (-jax.nn.log_sigmoid(real_logit).mean()
            - jax.nn.log_sigmoid(-fake_logit).mean())


def _cond_gen_loss(gp, dp, cond, z):
    fake = _mlp(gp, jnp.concatenate([z, cond], axis=1))
    fake_logit = _mlp(dp, jnp.concatenate([fake, cond], axis=1))[:, 0]
    return -jax.nn.log_sigmoid(fake_logit).mean()


class CTGANDebiaser:
    """A conditional tabular GAN that rebalances a biased dataset.

    A clean-room reimplementation of the *debiasing* idea from CTGAN
    (Xu et al., 2019) as used in arXiv:2603.18987 — written in JAX, with
    no dependency on the Business-Source-licensed ``sdv`` / ``ctgan``
    packages.

    The generator is **conditioned** on two discrete columns — the
    protected ``group`` and the binary ``outcome`` — and learns to
    produce realistic continuous feature columns for each
    ``(group, outcome)`` combination.  Debiasing then works exactly as
    CTGAN's training-by-sampling prescribes: :meth:`debias` synthesises
    a new dataset while sampling the *conditional distribution* in a
    rebalanced way — every group's favourable-outcome rate is set to the
    privileged group's rate — so the disparate-impact ratio of the
    debiased data moves toward 1.  The GAN's role is to keep the
    synthesised features realistic under that rebalanced conditioning.

    This redistributes disparity; as the paper stresses, it does not by
    itself remove structural bias without accompanying policy change.

    Parameters
    ----------
    latent_dim, hidden, seed
        As for :class:`SpatialGAN`.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64,
                 seed: int = 0):
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self._gp = None
        self._groups = None
        self._feature_cols = None
        self.history: list[float] = []

    def _cond(self, gi, oi):
        """One-hot ``(group, outcome)`` condition matrix."""
        ng = len(self._groups)
        cond = np.zeros((len(gi), ng + 2), dtype=np.float32)
        cond[np.arange(len(gi)), gi] = 1.0
        cond[np.arange(len(gi)), ng + oi] = 1.0
        return cond

    def fit(self, df, *, outcome_col, feature_cols, group_col="group",
            favorable=1, steps: int = 1500, batch_size: int = 128,
            lr: float = 2e-3):
        """Train the conditional GAN on a biased :class:`~pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
        outcome_col : str
            Binary outcome column (the value ``favorable`` is the
            favourable class).
        feature_cols : sequence of str
            Continuous feature columns the GAN learns to synthesise.
        group_col : str
            Protected-attribute column.
        favorable : default ``1``
            The favourable value of ``outcome_col``.
        """
        feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("need at least one feature column")
        self._groups = sorted(df[group_col].unique(), key=str)
        self._feature_cols = feature_cols
        self._group_col = group_col
        self._outcome_col = outcome_col
        self._favorable = favorable

        g_idx = {g: i for i, g in enumerate(self._groups)}
        gi = df[group_col].map(g_idx).to_numpy()
        oi = (df[outcome_col].to_numpy() == favorable).astype(int)
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        if feats.shape[0] < 2:
            raise ValueError("need at least two rows to fit")

        self._fmean = feats.mean(axis=0)
        self._fstd = feats.std(axis=0) + 1e-8
        std_feats = jnp.asarray((feats - self._fmean) / self._fstd)

        ng = len(self._groups)
        self._group_props = np.bincount(gi, minlength=ng) / len(gi)
        self._group_fav_rate = {
            g: float(oi[gi == i].mean()) if (gi == i).any() else 0.0
            for i, g in enumerate(self._groups)
        }
        cond = jnp.asarray(self._cond(gi, oi))
        cond_dim = ng + 2
        n_feat = std_feats.shape[1]
        n = std_feats.shape[0]

        key = jax.random.PRNGKey(self.seed)
        key, kg, kd = jax.random.split(key, 3)
        gp = _init_mlp(kg, [self.latent_dim + cond_dim, self.hidden,
                            self.hidden, n_feat])
        dp = _init_mlp(kd, [n_feat + cond_dim, self.hidden,
                            self.hidden, 1])
        gs = _adam_init(gp)
        ds = _adam_init(dp)

        @jax.jit
        def step(gp, dp, gs, ds, t, real, cnd, zd, zg):
            dl, dg = jax.value_and_grad(_cond_disc_loss)(
                dp, gp, real, cnd, zd)
            dp2, ds2 = _adam_step(dp, dg, ds, t, lr)
            gl, gg = jax.value_and_grad(_cond_gen_loss)(gp, dp2, cnd, zg)
            gp2, gs2 = _adam_step(gp, gg, gs, t, lr)
            return gp2, dp2, gs2, ds2, dl + gl

        bs = min(batch_size, n)
        self.history = []
        for t in range(1, int(steps) + 1):
            key, ks, kzd, kzg = jax.random.split(key, 4)
            idx = jax.random.randint(ks, (bs,), 0, n)
            real = std_feats[idx]
            cnd = cond[idx]
            zd = jax.random.normal(kzd, (bs, self.latent_dim))
            zg = jax.random.normal(kzg, (bs, self.latent_dim))
            gp, dp, gs, ds, loss = step(gp, dp, gs, ds, t, real, cnd, zd, zg)
            if t % 50 == 0:
                self.history.append(float(loss))
        self._gp = gp
        return self

    def debias(self, n: int, *, privileged, seed: int | None = None):
        """Synthesise ``n`` rebalanced rows as a :class:`~pandas.DataFrame`.

        Every group's favourable-outcome rate is set to the privileged
        group's rate, so the debiased dataset's disparate-impact ratio
        moves toward 1.  Feature columns are generated by the conditional
        GAN.

        Parameters
        ----------
        n : int
            Number of synthetic rows.
        privileged
            The group whose favourable-outcome rate the others are
            rebalanced to.
        seed : int, optional
            Sampling seed.

        Returns
        -------
        pandas.DataFrame
            Columns: the group column, the outcome column, and the
            feature columns — re-auditable with the morie.fairness
            metrics.
        """
        import pandas as pd

        if self._gp is None:
            raise RuntimeError("CTGANDebiaser is not fitted; call fit()")
        if privileged not in self._groups:
            raise ValueError(
                f"privileged group {privileged!r} not seen in training; "
                f"groups: {self._groups}"
            )
        target_rate = self._group_fav_rate[privileged]
        rng = np.random.default_rng(seed)
        gi = rng.choice(len(self._groups), size=int(n),
                        p=self._group_props)
        oi = (rng.random(int(n)) < target_rate).astype(int)
        cond = jnp.asarray(self._cond(gi, oi))

        key = jax.random.PRNGKey(self.seed if seed is None else int(seed))
        z = jax.random.normal(key, (int(n), self.latent_dim))
        std_feat = np.asarray(_mlp(self._gp, jnp.concatenate([z, cond],
                                                             axis=1)))
        feats = std_feat * self._fstd + self._fmean

        out = {
            self._group_col: [self._groups[i] for i in gi],
            self._outcome_col: np.where(oi == 1, self._favorable, 0),
        }
        for j, col in enumerate(self._feature_cols):
            out[col] = feats[:, j]
        return pd.DataFrame(out)
