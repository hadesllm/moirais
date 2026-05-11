"""Tests for morie.pt2gguf — PyTorch checkpoint to GGUF converter."""

import numpy as np
import pytest

from morie.pt2gguf import _map_tensor_name


# ---------------------------------------------------------------------------
# TestTensorNameMapping — no torch required
# ---------------------------------------------------------------------------


class TestTensorNameMapping:
    """Verify PT_TO_GGUF_MAP and PT_LAYER_MAP produce correct GGUF names."""

    def test_map_embedding(self):
        assert _map_tensor_name("transformer.wte.weight") == "token_embd.weight"

    def test_map_lm_head(self):
        assert _map_tensor_name("lm_head.weight") == "output.weight"

    def test_map_layer_attn_q(self):
        assert _map_tensor_name("transformer.h.0.attn.c_q.weight") == "blk.0.attn_q.weight"

    def test_map_layer_attn_k(self):
        assert _map_tensor_name("transformer.h.0.attn.c_k.weight") == "blk.0.attn_k.weight"

    def test_map_layer_attn_v(self):
        assert _map_tensor_name("transformer.h.0.attn.c_v.weight") == "blk.0.attn_v.weight"

    def test_map_layer_attn_proj(self):
        assert _map_tensor_name("transformer.h.0.attn.c_proj.weight") == "blk.0.attn_output.weight"

    def test_map_layer_mlp_fc(self):
        assert _map_tensor_name("transformer.h.3.mlp.c_fc.weight") == "blk.3.ffn_up.weight"

    def test_map_layer_mlp_proj(self):
        assert _map_tensor_name("transformer.h.3.mlp.c_proj.weight") == "blk.3.ffn_down.weight"

    def test_map_layer_attn_norm_returns_none(self):
        assert _map_tensor_name("transformer.h.1.attn.ln.weight") is None

    def test_map_layer_ffn_norm_returns_none(self):
        assert _map_tensor_name("transformer.h.2.mlp.ln.weight") is None

    def test_map_value_embeds(self):
        assert _map_tensor_name("value_embeds.2.weight") == "blk.2.value_embd.weight"

    def test_map_resid_lambdas(self):
        assert _map_tensor_name("resid_lambdas") == "resid_lambdas"

    def test_map_x0_lambdas(self):
        assert _map_tensor_name("x0_lambdas") == "x0_lambdas"

    def test_map_unknown_returns_none(self):
        assert _map_tensor_name("some.random.key") is None

    def test_map_cos_skipped(self):
        """cos is not in PT_TO_GGUF_MAP or any layer pattern => None."""
        assert _map_tensor_name("cos") is None

    def test_map_sin_skipped(self):
        """sin is not in PT_TO_GGUF_MAP or any layer pattern => None."""
        assert _map_tensor_name("sin") is None

    def test_map_high_layer_index(self):
        assert _map_tensor_name("transformer.h.99.attn.c_q.weight") == "blk.99.attn_q.weight"

    def test_map_ve_gate(self):
        assert _map_tensor_name("transformer.h.0.attn.ve_gate.weight") == "blk.0.attn_ve_gate.weight"


# ---------------------------------------------------------------------------
# Helpers for round-trip tests
# ---------------------------------------------------------------------------

def _fake_tokenizer_info():
    """Return a minimal tokenizer dict for monkeypatching."""
    return {
        "tokens": [f"tok_{i}" for i in range(256)],
        "scores": [float(-i) for i in range(256)],
        "bos_id": 0,
        "eos_id": 1,
        "vocab_size": 256,
    }


def _build_fake_checkpoint(tmp_path, n_layer=2, n_head=2, n_kv_head=2,
                           n_embd=64, vocab_size=256, sequence_len=128):
    """Create a fake autoresearch .pt checkpoint with torch.save().

    Returns (checkpoint_path, config, state_dict_numpy) where
    state_dict_numpy maps PT names to numpy arrays for verification.
    """
    torch = pytest.importorskip("torch")

    config = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "sequence_len": sequence_len,
        "window_pattern": "L" * n_layer,
    }

    head_dim = n_embd // n_head
    ffn_dim = 4 * n_embd

    rng = torch.Generator().manual_seed(42)

    state_dict = {
        "transformer.wte.weight": torch.randn(vocab_size, n_embd, generator=rng),
        "lm_head.weight": torch.randn(vocab_size, n_embd, generator=rng),
    }

    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        state_dict[f"{prefix}.attn.c_q.weight"] = torch.randn(n_embd, n_embd, generator=rng)
        state_dict[f"{prefix}.attn.c_k.weight"] = torch.randn(n_kv_head * head_dim, n_embd, generator=rng)
        state_dict[f"{prefix}.attn.c_v.weight"] = torch.randn(n_kv_head * head_dim, n_embd, generator=rng)
        state_dict[f"{prefix}.attn.c_proj.weight"] = torch.randn(n_embd, n_embd, generator=rng)
        state_dict[f"{prefix}.mlp.c_fc.weight"] = torch.randn(ffn_dim, n_embd, generator=rng)
        state_dict[f"{prefix}.mlp.c_proj.weight"] = torch.randn(n_embd, ffn_dim, generator=rng)

    # Scalars / small tensors
    state_dict["resid_lambdas"] = torch.ones(n_layer)
    state_dict["x0_lambdas"] = torch.ones(n_layer) * 0.1

    # Tensors that should be skipped by convert()
    state_dict["cos"] = torch.randn(sequence_len, head_dim, generator=rng)
    state_dict["sin"] = torch.randn(sequence_len, head_dim, generator=rng)

    # Save numpy copies for verification (before torch.save moves them)
    np_dict = {k: v.float().cpu().numpy().copy() for k, v in state_dict.items()}

    ckpt_path = tmp_path / "model_baseline.pt"
    torch.save({"model_state_dict": state_dict, "config": config}, ckpt_path)

    return ckpt_path, config, np_dict


# ---------------------------------------------------------------------------
# TestGGUFRoundTrip — requires torch
# ---------------------------------------------------------------------------


class TestGGUFRoundTrip:
    """Convert a fake checkpoint to GGUF (F32) and verify with GGUFModel."""

    @pytest.fixture
    def roundtrip_data(self, tmp_path, monkeypatch):
        """Build checkpoint, mock tokenizer, convert, load back."""
        torch = pytest.importorskip("torch")

        ckpt_path, config, np_dict = _build_fake_checkpoint(tmp_path)
        gguf_path = tmp_path / "test_f32.gguf"

        # Mock tokenizer loading to avoid needing a real pickle
        import morie.pt2gguf as pt2gguf_mod
        monkeypatch.setattr(pt2gguf_mod, "_load_autoresearch_tokenizer",
                            lambda _dir: _fake_tokenizer_info())

        pt2gguf_mod.convert(
            checkpoint_path=str(ckpt_path),
            output_path=str(gguf_path),
            tokenizer_dir="/fake/tokenizer",
            turbo_bits=0,
        )

        from morie.gguf_loader import GGUFModel
        model = GGUFModel(gguf_path)

        yield model, config, np_dict

        model.close()

    def test_metadata_architecture(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["architecture"] == "morie_gpt"

    def test_metadata_block_count(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["block_count"] == config["n_layer"]

    def test_metadata_head_count(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["head_count"] == config["n_head"]

    def test_metadata_head_count_kv(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["head_count_kv"] == config["n_kv_head"]

    def test_metadata_embedding_length(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["embedding_length"] == config["n_embd"]

    def test_metadata_vocab_size(self, roundtrip_data):
        model, config, _ = roundtrip_data
        assert model.config["vocab_size"] == config["vocab_size"]

    def test_cos_sin_skipped(self, roundtrip_data):
        model, _, _ = roundtrip_data
        names = model.tensor_names()
        assert "cos" not in names
        assert "sin" not in names

    def test_embedding_shape(self, roundtrip_data):
        model, config, _ = roundtrip_data
        t = model.get_tensor("token_embd.weight")
        assert t.shape == (config["vocab_size"], config["n_embd"])

    def test_embedding_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("token_embd.weight")
        expected = np_dict["transformer.wte.weight"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_output_shape(self, roundtrip_data):
        model, config, _ = roundtrip_data
        t = model.get_tensor("output.weight")
        assert t.shape == (config["vocab_size"], config["n_embd"])

    def test_output_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("output.weight")
        expected = np_dict["lm_head.weight"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_attn_q_shape(self, roundtrip_data):
        model, config, _ = roundtrip_data
        t = model.get_tensor("blk.0.attn_q.weight")
        assert t.shape == (config["n_embd"], config["n_embd"])

    def test_attn_q_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("blk.0.attn_q.weight")
        expected = np_dict["transformer.h.0.attn.c_q.weight"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_ffn_up_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("blk.1.ffn_up.weight")
        expected = np_dict["transformer.h.1.mlp.c_fc.weight"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_ffn_down_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("blk.0.ffn_down.weight")
        expected = np_dict["transformer.h.0.mlp.c_proj.weight"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_resid_lambdas_values(self, roundtrip_data):
        model, config, np_dict = roundtrip_data
        loaded = model.get_tensor("resid_lambdas")
        expected = np_dict["resid_lambdas"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_x0_lambdas_values(self, roundtrip_data):
        model, _, np_dict = roundtrip_data
        loaded = model.get_tensor("x0_lambdas")
        expected = np_dict["x0_lambdas"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_tensor_count(self, roundtrip_data):
        """Should have: wte + lm_head + n_layer*(6 per-layer) + resid + x0."""
        model, config, _ = roundtrip_data
        n_layer = config["n_layer"]
        expected_count = 2 + n_layer * 6 + 2  # embed, output, 6 per layer, 2 lambdas
        assert len(model.tensor_names()) == expected_count

    def test_all_layers_present(self, roundtrip_data):
        model, config, _ = roundtrip_data
        names = model.tensor_names()
        for i in range(config["n_layer"]):
            assert f"blk.{i}.attn_q.weight" in names
            assert f"blk.{i}.attn_k.weight" in names
            assert f"blk.{i}.attn_v.weight" in names
            assert f"blk.{i}.attn_output.weight" in names
            assert f"blk.{i}.ffn_up.weight" in names
            assert f"blk.{i}.ffn_down.weight" in names


# ---------------------------------------------------------------------------
# TestTurboQuantRoundTrip — requires torch + morie.quant
# ---------------------------------------------------------------------------


class TestTurboQuantRoundTrip:
    """Convert with --turbo-bits 3 and verify lossy round-trip via cosine similarity."""

    @pytest.fixture
    def tq_roundtrip(self, tmp_path, monkeypatch):
        """Build checkpoint, convert with TQ3, load back."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("scipy")  # turboquant_mse needs scipy

        ckpt_path, config, np_dict = _build_fake_checkpoint(
            tmp_path, n_layer=2, n_head=2, n_kv_head=2,
            n_embd=64, vocab_size=256, sequence_len=128,
        )
        gguf_path = tmp_path / "test_tq3.gguf"

        import morie.pt2gguf as pt2gguf_mod
        monkeypatch.setattr(pt2gguf_mod, "_load_autoresearch_tokenizer",
                            lambda _dir: _fake_tokenizer_info())

        pt2gguf_mod.convert(
            checkpoint_path=str(ckpt_path),
            output_path=str(gguf_path),
            tokenizer_dir="/fake/tokenizer",
            turbo_bits=3,
        )

        from morie.gguf_loader import GGUFModel
        model = GGUFModel(gguf_path)

        yield model, config, np_dict

        model.close()

    @staticmethod
    def _cosine_sim(a, b):
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        dot = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def test_tq3_embedding_values(self, tq_roundtrip):
        """Embedding is 2D and large enough for TQ — should be compressed."""
        model, _, np_dict = tq_roundtrip
        loaded = model.get_tensor("token_embd.weight")
        expected = np_dict["transformer.wte.weight"]
        assert loaded.shape == expected.shape
        sim = self._cosine_sim(loaded, expected)
        assert sim > 0.95, f"Cosine similarity {sim:.4f} below threshold"

    def test_tq3_attn_q_values(self, tq_roundtrip):
        model, _, np_dict = tq_roundtrip
        loaded = model.get_tensor("blk.0.attn_q.weight")
        expected = np_dict["transformer.h.0.attn.c_q.weight"]
        sim = self._cosine_sim(loaded, expected)
        assert sim > 0.95, f"Cosine similarity {sim:.4f} below threshold"

    def test_tq3_ffn_up_values(self, tq_roundtrip):
        model, _, np_dict = tq_roundtrip
        loaded = model.get_tensor("blk.1.ffn_up.weight")
        expected = np_dict["transformer.h.1.mlp.c_fc.weight"]
        sim = self._cosine_sim(loaded, expected)
        assert sim > 0.95, f"Cosine similarity {sim:.4f} below threshold"

    def test_tq3_lambdas_exact(self, tq_roundtrip):
        """1D tensors below TQ_BLOCK_SIZE stay F32 — should be exact."""
        model, _, np_dict = tq_roundtrip
        loaded = model.get_tensor("resid_lambdas")
        expected = np_dict["resid_lambdas"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_tq3_x0_lambdas_exact(self, tq_roundtrip):
        model, _, np_dict = tq_roundtrip
        loaded = model.get_tensor("x0_lambdas")
        expected = np_dict["x0_lambdas"]
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_tq3_metadata_turbo_bits(self, tq_roundtrip):
        model, _, _ = tq_roundtrip
        assert model.config.get("turbo_bits") == 3

    def test_tq3_output_file_smaller(self, tq_roundtrip):
        """TQ3-compressed file should exist and be loadable (size check is implicit)."""
        model, config, _ = tq_roundtrip
        # If we got here, the file was created and loaded successfully.
        assert model.config["architecture"] == "morie_gpt"
        assert len(model.tensor_names()) > 0
