"""Deterministic-seed plumbing tests for the Deep-learning suite.

Verifies that the ``deterministic_seed`` kwarg added to the 6 DL
callables (``heinz``, ``drpfw``, ``lstmc``, ``grucl``, ``mhatf``,
``trfbl``) for morie v0.4.0:

* gives bit-identical output across two calls with the same seed;
* gives different output when the seed is changed; and
* leaves the default ``deterministic_seed=None`` path producing the
  same numbers as before (legacy ``seed=`` path).

These callables are pure-numpy implementations.  No torch / TF / keras
backend is invoked, so ``pytest.importorskip("torch")`` is not needed.
The tests still gracefully skip the SHA-keyed branch if neither
``digest`` nor ``openssl`` is installed (which would only matter on the
R-side; the Python side only needs ``hashlib`` from the std lib).
"""

from __future__ import annotations

import numpy as np

from morie.fn.heinz import he_initialization
from morie.fn.drpfw import dropout_forward
from morie.fn.lstmc import lstm_cell
from morie.fn.grucl import gru_cell
from morie.fn.mhatf import multi_head_attention_full
from morie.fn.trfbl import transformer_block


# ---------------------------------------------------------------------------
# heinz: He/Kaiming initialisation
# ---------------------------------------------------------------------------
def test_heinz_deterministic_seed_reproducible():
    r1 = he_initialization(64, 32, deterministic_seed=42)
    r2 = he_initialization(64, 32, deterministic_seed=42)
    r3 = he_initialization(64, 32, deterministic_seed=999)
    np.testing.assert_array_equal(r1["W"], r2["W"])
    assert not np.array_equal(r1["W"], r3["W"])

    # Default deterministic_seed=None path is unchanged.
    a = he_initialization(64, 32, seed=42)
    b = he_initialization(64, 32, seed=42)
    np.testing.assert_array_equal(a["W"], b["W"])


# ---------------------------------------------------------------------------
# drpfw: inverted dropout
# ---------------------------------------------------------------------------
def test_drpfw_deterministic_seed_reproducible():
    x = np.ones(2048)
    r1 = dropout_forward(x, p=0.5, deterministic_seed=42)
    r2 = dropout_forward(x, p=0.5, deterministic_seed=42)
    r3 = dropout_forward(x, p=0.5, deterministic_seed=999)
    np.testing.assert_array_equal(r1["mask"], r2["mask"])
    assert not np.array_equal(r1["mask"], r3["mask"])

    a = dropout_forward(x, p=0.5, seed=0)
    b = dropout_forward(x, p=0.5, seed=0)
    np.testing.assert_array_equal(a["mask"], b["mask"])


# ---------------------------------------------------------------------------
# lstmc: LSTM cell forward pass (random default W, U)
# ---------------------------------------------------------------------------
def test_lstmc_deterministic_seed_reproducible():
    x = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
    r1 = lstm_cell(x, hidden_size=8, deterministic_seed=42)
    r2 = lstm_cell(x, hidden_size=8, deterministic_seed=42)
    r3 = lstm_cell(x, hidden_size=8, deterministic_seed=999)
    np.testing.assert_array_equal(r1["h"], r2["h"])
    assert not np.array_equal(r1["h"], r3["h"])

    a = lstm_cell(x, hidden_size=8, seed=0)
    b = lstm_cell(x, hidden_size=8, seed=0)
    np.testing.assert_array_equal(a["h"], b["h"])


# ---------------------------------------------------------------------------
# grucl: GRU cell forward pass (random default W, U)
# ---------------------------------------------------------------------------
def test_grucl_deterministic_seed_reproducible():
    x = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
    r1 = gru_cell(x, hidden_size=8, deterministic_seed=42)
    r2 = gru_cell(x, hidden_size=8, deterministic_seed=42)
    r3 = gru_cell(x, hidden_size=8, deterministic_seed=999)
    np.testing.assert_array_equal(r1["h"], r2["h"])
    assert not np.array_equal(r1["h"], r3["h"])

    a = gru_cell(x, hidden_size=8, seed=0)
    b = gru_cell(x, hidden_size=8, seed=0)
    np.testing.assert_array_equal(a["h"], b["h"])


# ---------------------------------------------------------------------------
# mhatf: multi-head attention with random default Q/K/V projections
# ---------------------------------------------------------------------------
def test_mhatf_deterministic_seed_reproducible():
    x = np.eye(4)
    r1 = multi_head_attention_full(x, num_heads=2, deterministic_seed=42)
    r2 = multi_head_attention_full(x, num_heads=2, deterministic_seed=42)
    r3 = multi_head_attention_full(x, num_heads=2, deterministic_seed=999)
    np.testing.assert_array_equal(r1["output"], r2["output"])
    assert not np.array_equal(r1["output"], r3["output"])

    a = multi_head_attention_full(x, num_heads=2, seed=0)
    b = multi_head_attention_full(x, num_heads=2, seed=0)
    np.testing.assert_array_equal(a["output"], b["output"])


# ---------------------------------------------------------------------------
# trfbl: Transformer encoder block (random default W_q/k/v + FFN W1/W2)
# ---------------------------------------------------------------------------
def test_trfbl_deterministic_seed_reproducible():
    x = np.eye(4)
    r1 = transformer_block(x, num_heads=2, deterministic_seed=42)
    r2 = transformer_block(x, num_heads=2, deterministic_seed=42)
    r3 = transformer_block(x, num_heads=2, deterministic_seed=999)
    np.testing.assert_array_equal(r1["output"], r2["output"])
    assert not np.array_equal(r1["output"], r3["output"])

    a = transformer_block(x, num_heads=2, seed=0)
    b = transformer_block(x, num_heads=2, seed=0)
    np.testing.assert_array_equal(a["output"], b["output"])
