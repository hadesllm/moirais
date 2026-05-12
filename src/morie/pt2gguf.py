"""
Convert autoresearch PyTorch checkpoints to GGUF format.

Bridges autoresearch's train.py GPT architecture to MORIE's inference engine.
Supports optional TurboQuant weight compression during conversion.

Usage:
    python -m morie.pt2gguf --checkpoint model_baseline.pt --output morie_model.gguf
    python -m morie.pt2gguf --checkpoint model_baseline.pt --output morie_3bit.gguf --turbo-bits 3
"""

import argparse
import io
import os
import pickle
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_TQ2 = 100
GGML_TYPE_TQ3 = 101
GGML_TYPE_TQ4 = 102

GGUFValueType_UINT8 = 0
GGUFValueType_INT8 = 1
GGUFValueType_UINT16 = 2
GGUFValueType_INT16 = 3
GGUFValueType_UINT32 = 4
GGUFValueType_INT32 = 5
GGUFValueType_FLOAT32 = 6
GGUFValueType_BOOL = 7
GGUFValueType_STRING = 8
GGUFValueType_ARRAY = 9
GGUFValueType_UINT64 = 10
GGUFValueType_INT64 = 11
GGUFValueType_FLOAT64 = 12

TQ_BLOCK_SIZE = 256

PT_TO_GGUF_MAP = {
    "transformer.wte.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
}

PT_LAYER_MAP = {
    "attn.c_q.weight": "attn_q.weight",
    "attn.c_k.weight": "attn_k.weight",
    "attn.c_v.weight": "attn_v.weight",
    "attn.c_proj.weight": "attn_output.weight",
    "mlp.c_fc.weight": "ffn_up.weight",
    "mlp.c_proj.weight": "ffn_down.weight",
    "attn.ve_gate.weight": "attn_ve_gate.weight",
}


def _map_tensor_name(pt_name):
    if pt_name in PT_TO_GGUF_MAP:
        return PT_TO_GGUF_MAP[pt_name]

    if pt_name.startswith("transformer.h."):
        parts = pt_name.split(".", 3)
        layer_idx = parts[2]
        suffix = parts[3]
        if suffix in PT_LAYER_MAP:
            return f"blk.{layer_idx}.{PT_LAYER_MAP[suffix]}"

    if pt_name.startswith("value_embeds."):
        parts = pt_name.split(".")
        layer_idx = parts[1]
        return f"blk.{layer_idx}.value_embd.weight"

    if pt_name in ("resid_lambdas", "x0_lambdas"):
        return pt_name

    return None


def _write_string(buf, s):
    encoded = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(encoded)))
    buf.write(encoded)


def _write_kv_string(buf, key, value):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_STRING))
    _write_string(buf, value)


def _write_kv_uint32(buf, key, value):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_UINT32))
    buf.write(struct.pack("<I", value))


def _write_kv_uint64(buf, key, value):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_UINT64))
    buf.write(struct.pack("<Q", value))


def _write_kv_float32(buf, key, value):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_FLOAT32))
    buf.write(struct.pack("<f", value))


def _write_kv_bool(buf, key, value):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_BOOL))
    buf.write(struct.pack("<B", 1 if value else 0))


def _write_kv_array_string(buf, key, values):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_ARRAY))
    buf.write(struct.pack("<I", GGUFValueType_STRING))
    buf.write(struct.pack("<Q", len(values)))
    for v in values:
        _write_string(buf, v)


def _write_kv_array_float32(buf, key, values):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_ARRAY))
    buf.write(struct.pack("<I", GGUFValueType_FLOAT32))
    buf.write(struct.pack("<Q", len(values)))
    for v in values:
        buf.write(struct.pack("<f", v))


def _write_kv_array_int32(buf, key, values):
    _write_string(buf, key)
    buf.write(struct.pack("<I", GGUFValueType_ARRAY))
    buf.write(struct.pack("<I", GGUFValueType_INT32))
    buf.write(struct.pack("<Q", len(values)))
    for v in values:
        buf.write(struct.pack("<i", v))


def _pad_to_alignment(buf, alignment=32):
    pos = buf.tell()
    pad = (alignment - pos % alignment) % alignment
    buf.write(b"\x00" * pad)


def _load_autoresearch_tokenizer(tokenizer_dir):
    tok_path = Path(tokenizer_dir) / "tokenizer.pkl"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    with open(tok_path, "rb") as f:
        enc = pickle.load(f)  # noqa: S301 -- local autoresearch tokenizer only, not user input

    tokens = []
    scores = []
    for i in range(enc.n_vocab):
        try:
            tok_bytes = enc.decode_single_token_bytes(i)
            tokens.append(tok_bytes.decode("utf-8", errors="replace"))
        except Exception:
            tokens.append(f"<|token_{i}|>")
        scores.append(float(-i))

    bos_id = None
    eos_id = None
    for name, tid in enc._special_tokens.items():
        if "reserved_0" in name:
            bos_id = tid
        if "reserved_1" in name:
            eos_id = tid

    return {
        "tokens": tokens,
        "scores": scores,
        "bos_id": bos_id or 0,
        "eos_id": eos_id or 0,
        "vocab_size": enc.n_vocab,
    }


def _turbo_compress_tensor(tensor_np, bits):
    from morie.quant import turboquant_mse

    flat = tensor_np.flatten().astype(np.float64)
    n = len(flat)
    pad = (TQ_BLOCK_SIZE - n % TQ_BLOCK_SIZE) % TQ_BLOCK_SIZE
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad)])

    n_blocks = len(flat) // TQ_BLOCK_SIZE
    packed = io.BytesIO()

    packed.write(struct.pack("<I", n))
    packed.write(struct.pack("<I", n_blocks))
    packed.write(struct.pack("<I", bits))

    for i in range(n_blocks):
        chunk = flat[i * TQ_BLOCK_SIZE : (i + 1) * TQ_BLOCK_SIZE]
        block = turboquant_mse(chunk, bits=bits, rotation_seed=42 + i)
        packed.write(struct.pack("<f", float(block.norm)))
        packed.write(struct.pack("<H", block.rotation_seed & 0xFFFF))
        idx_bytes = np.packbits(
            np.unpackbits(block.indices.astype(np.uint8), bitorder="little")[: TQ_BLOCK_SIZE * bits], bitorder="little"
        )
        packed.write(idx_bytes.tobytes())

    return packed.getvalue()


def convert(checkpoint_path, output_path, tokenizer_dir=None, turbo_bits=0):
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except ImportError:
        raise RuntimeError("PyTorch required for checkpoint loading")

    config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    n_layer = config["n_layer"]
    n_head = config["n_head"]
    n_kv_head = config.get("n_kv_head", n_head)
    n_embd = config["n_embd"]
    vocab_size = config["vocab_size"]
    seq_len = config.get("sequence_len", 2048)
    window_pattern = config.get("window_pattern", "L" * n_layer)

    head_dim = n_embd // n_head
    ffn_dim = 4 * n_embd

    if tokenizer_dir is None:
        tokenizer_dir = os.path.expanduser("~/.cache/autoresearch/tokenizer")
    tok_info = _load_autoresearch_tokenizer(tokenizer_dir)

    has_value_embeds = any(k.startswith("value_embeds.") for k in state_dict)
    has_qk_norm = any("attn.ln" in k for k in state_dict)
    has_resid_lambdas = "resid_lambdas" in state_dict

    tensors = {}
    skipped = []
    for pt_name, pt_tensor in state_dict.items():
        if pt_name in ("cos", "sin"):
            skipped.append(pt_name)
            continue

        gguf_name = _map_tensor_name(pt_name)
        if gguf_name is None:
            skipped.append(pt_name)
            continue

        arr = pt_tensor.float().cpu().numpy()
        tensors[gguf_name] = arr

    if skipped:
        print(f"  Skipped {len(skipped)} tensors: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    print(f"  Mapped {len(tensors)} tensors")

    arch = "morie_gpt"

    metadata_buf = io.BytesIO()
    kv_count = 0

    def kv_string(k, v):
        nonlocal kv_count
        _write_kv_string(metadata_buf, k, v)
        kv_count += 1

    def kv_uint32(k, v):
        nonlocal kv_count
        _write_kv_uint32(metadata_buf, k, v)
        kv_count += 1

    def kv_float32(k, v):
        nonlocal kv_count
        _write_kv_float32(metadata_buf, k, v)
        kv_count += 1

    def kv_bool(k, v):
        nonlocal kv_count
        _write_kv_bool(metadata_buf, k, v)
        kv_count += 1

    def kv_arr_str(k, v):
        nonlocal kv_count
        _write_kv_array_string(metadata_buf, k, v)
        kv_count += 1

    def kv_arr_f32(k, v):
        nonlocal kv_count
        _write_kv_array_float32(metadata_buf, k, v)
        kv_count += 1

    kv_string("general.architecture", arch)
    kv_string("general.name", "morie-autoresearch-gpt")
    kv_uint32(f"{arch}.context_length", seq_len)
    kv_uint32(f"{arch}.block_count", n_layer)
    kv_uint32(f"{arch}.attention.head_count", n_head)
    kv_uint32(f"{arch}.attention.head_count_kv", n_kv_head)
    kv_uint32(f"{arch}.embedding_length", n_embd)
    kv_uint32(f"{arch}.feed_forward_length", ffn_dim)
    kv_uint32(f"{arch}.vocab_size", vocab_size)
    kv_uint32(f"{arch}.rope.dimension_count", head_dim)
    kv_float32(f"{arch}.rope.freq_base", 10000.0)
    kv_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    kv_string(f"{arch}.window_pattern", window_pattern)
    kv_float32(f"{arch}.logit_softcap", 15.0)
    kv_bool(f"{arch}.has_value_embeds", has_value_embeds)
    kv_bool(f"{arch}.has_qk_norm", has_qk_norm)
    kv_bool(f"{arch}.has_resid_lambdas", has_resid_lambdas)
    kv_string(f"{arch}.ffn_activation", "relu_squared")

    if turbo_bits > 0:
        kv_uint32(f"{arch}.turbo_bits", turbo_bits)

    kv_string("tokenizer.ggml.model", "bpe")
    kv_arr_str("tokenizer.ggml.tokens", tok_info["tokens"])
    kv_arr_f32("tokenizer.ggml.scores", tok_info["scores"])
    kv_uint32("tokenizer.ggml.bos_token_id", tok_info["bos_id"])
    kv_uint32("tokenizer.ggml.eos_token_id", tok_info["eos_id"])

    metadata_bytes = metadata_buf.getvalue()

    tq_type_map = {2: GGML_TYPE_TQ2, 3: GGML_TYPE_TQ3, 4: GGML_TYPE_TQ4}
    tensor_entries = []
    tensor_data_parts = []
    data_offset = 0

    for name, arr in tensors.items():
        use_tq = turbo_bits > 0 and arr.ndim >= 2 and arr.size >= TQ_BLOCK_SIZE

        if use_tq:
            compressed = _turbo_compress_tensor(arr, turbo_bits)
            dtype = tq_type_map[turbo_bits]
            tensor_data_parts.append(compressed)
            data_size = len(compressed)
        else:
            raw = arr.astype(np.float32).tobytes()
            dtype = GGML_TYPE_F32
            tensor_data_parts.append(raw)
            data_size = len(raw)

        tensor_entries.append(
            {
                "name": name,
                "ndims": arr.ndim,
                "shape": list(arr.shape),
                "dtype": dtype,
                "offset": data_offset,
            }
        )

        padded_size = data_size + (32 - data_size % 32) % 32
        data_offset += padded_size

    out = io.BytesIO()

    out.write(struct.pack("<I", GGUF_MAGIC))
    out.write(struct.pack("<I", GGUF_VERSION))
    out.write(struct.pack("<Q", len(tensor_entries)))
    out.write(struct.pack("<Q", kv_count))

    out.write(metadata_bytes)

    for entry in tensor_entries:
        _write_string(out, entry["name"])
        out.write(struct.pack("<I", entry["ndims"]))
        for dim in entry["shape"]:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<I", entry["dtype"]))
        out.write(struct.pack("<Q", entry["offset"]))

    _pad_to_alignment(out, 32)

    for i, data in enumerate(tensor_data_parts):
        out.write(data)
        pad = (32 - len(data) % 32) % 32
        out.write(b"\x00" * pad)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(out.getvalue())

    file_size = output_path.stat().st_size
    print(f"  Written: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Tensors: {len(tensor_entries)}")
    print(f"  Architecture: {arch}")
    if turbo_bits > 0:
        n_compressed = sum(1 for e in tensor_entries if e["dtype"] != GGML_TYPE_F32)
        print(f"  TurboQuant: {turbo_bits}-bit ({n_compressed} tensors compressed)")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert autoresearch .pt checkpoint to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to model_baseline.pt")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Autoresearch tokenizer directory (default: ~/.cache/autoresearch/tokenizer)",
    )
    parser.add_argument(
        "--turbo-bits",
        type=int,
        default=0,
        choices=[0, 2, 3, 4],
        help="TurboQuant compression bits (0=none, 2/3/4=compress)",
    )
    args = parser.parse_args()

    print("Converting autoresearch checkpoint to GGUF...")
    convert(args.checkpoint, args.output, args.tokenizer_dir, args.turbo_bits)
    print("Done.")
