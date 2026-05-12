# SPDX-License-Identifier: GPL-2.0-only
"""Smoke tests for the morie.fn LLM-architecture suite (20 callables)."""

from __future__ import annotations

import numpy as np
import pytest

from morie.fn import bpblm, cslat, cslnc, grdcl, grpqa, kvcmp, lradw
from morie.fn import moeml, pplxm, rlhfd, rmsnr, rptpn, spqkv, swigl
from morie.fn import tknbp, tmpsc, topkd, toppd, wdemb
from morie.fn import flshA  # noqa: N812  (matches spec short name)


# ───────────────────────── 20 unit checks ──────────────────────────

def test_tknbp():
    r = tknbp.bpe_tokenizer(
        ["low", "low", "lower", "newest", "newest", "newest"],
        num_merges=3,
    )
    assert r["n_merges"] == 3
    assert len(r["vocab"]) > 0


def test_wdemb_identity_lookup():
    E = np.eye(4)
    r = wdemb.word_embedding([0, 2], E=E)
    assert np.allclose(r["tensor"], E[[0, 2]])


def test_cslat_shape_and_inf():
    r = cslat.causal_attention_mask(3)
    M = r["tensor"]
    assert M.shape == (3, 3)
    assert np.isinf(M[0, 1]) and M[0, 1] < 0
    assert M[1, 0] == 0


def test_grpqa_shape():
    Q = np.zeros((4, 2, 8)); K = V = np.zeros((2, 2, 8))
    r = grpqa.grouped_query_attention(Q, K, V, n_heads=4, n_kv_heads=2)
    assert r["tensor"].shape == (4, 2, 8)


def test_swigl_silu_at_zero():
    x = np.zeros((1, 4)); W = V = np.eye(4)
    r = swigl.swiglu_activation(x, W=W, V=V)
    assert r["tensor"].shape == (1, 4)
    # SiLU(0)=0 -> output is all-zero.
    assert np.allclose(r["tensor"], 0.0)


def test_rmsnr_unit_norm_after_normalising():
    x = np.array([[3.0, 4.0]])
    r = rmsnr.rms_norm(x, eps=0.0)
    assert np.allclose(r["tensor"], x / np.sqrt(12.5))


def test_kvcmp_append_grows_T():
    K = np.zeros((2, 4)); V = np.zeros((2, 4))
    r = kvcmp.kv_cache_management(K, V, np.ones((1, 4)), np.ones((1, 4)))
    assert r["T"] == 3


def test_tmpsc_sums_to_one():
    r = tmpsc.temperature_scaling([1.0, 2.0, 3.0], T=1.0)
    assert np.isclose(r["tensor"].sum(), 1.0)


def test_topkd_keeps_exactly_k():
    r = topkd.top_k_decoding([1.0, 2.0, 3.0, 4.0, 5.0], k=2)
    assert int((r["tensor"] > 0).sum()) == 2


def test_toppd_keeps_minimal_set():
    r = toppd.top_p_nucleus([0.0, 0.0, 5.0], p=0.5)
    assert int(r["n_kept"]) == 1


def test_rptpn_asymmetric():
    r = rptpn.repetition_penalty([2.0, -2.0, 1.0],
                                 generated=[0, 1], alpha=2.0)
    assert np.allclose(r["tensor"], [1.0, -4.0, 1.0])


def test_pplxm_uniform_binary():
    r = pplxm.perplexity_metric([np.log(0.5), np.log(0.5)])
    assert round(float(r["value"]), 4) == 2.0


def test_bpblm_one_bit_per_byte():
    r = bpblm.bits_per_byte([np.log(2.0)] * 4, n_bytes=4)
    assert np.isclose(float(r["value"]), 1.0)


def test_cslnc_at_zero_is_lr_max():
    r = cslnc.cosine_lr_schedule(0, lr_max=1.0, lr_min=0.0,
                                  total_steps=10, warmup_steps=0)
    assert np.isclose(float(r["value"]), 1.0)


def test_grdcl_clips_to_max_norm():
    r = grdcl.gradient_clipping([3.0, 4.0], max_norm=1.0)
    assert np.isclose(np.linalg.norm(r["tensor"]), 1.0)


def test_lradw_half_at_half_warmup():
    r = lradw.lr_warmup(500, lr_target=1.0, warmup_steps=1000)
    assert np.isclose(float(r["value"]), 0.5)


def test_flshA_matches_naive():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((6, 4))
    K = rng.standard_normal((6, 4))
    V = rng.standard_normal((6, 4))
    r = flshA.flash_attention(Q, K, V, block_size=2)
    s = Q @ K.T / np.sqrt(4)
    p = np.exp(s - s.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    assert np.allclose(r["tensor"], p @ V, atol=1e-10)


def test_spqkv_diagonal_always_on():
    r = spqkv.sparse_attention(8, window=1, stride=4, n_random=0)
    assert all(r["boolean"][i, i] for i in range(8))


def test_moeml_routes_to_argmax_expert():
    x = np.array([[1.0, 0.0]])
    W_gate = np.array([[10.0, -10.0], [0.0, 0.0]])
    r = moeml.mixture_of_experts(x, W_gate=W_gate, top_k=1)
    assert int(r["topk_idx"][0, 0]) == 0


def test_rlhfd_linear_combination():
    r = rlhfd.rlhf_reward(np.array([[1.0, 1.0]]),
                          w=np.array([0.5, 0.5]), b=0.0)
    assert float(r["value"]) == 1.0


# ──────────────────── headline mass property checks ────────────────

def test_tmpsc_topkd_toppd_all_sum_to_one():
    z = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
    assert np.isclose(tmpsc.temperature_scaling(z, T=0.7)["tensor"].sum(), 1.0)
    assert np.isclose(topkd.top_k_decoding(z, k=3)["tensor"].sum(), 1.0)
    assert np.isclose(toppd.top_p_nucleus(z, p=0.9)["tensor"].sum(), 1.0)
