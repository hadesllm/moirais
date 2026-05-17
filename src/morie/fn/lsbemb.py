# morie.fn -- function file (hadesllm/morie)
"""Embed a binary message into integer data using LSB steganography."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lsb_embed(
    cover: np.ndarray,
    message_bits: np.ndarray,
    *,
    n_lsb: int = 1,
) -> DescriptiveResult:
    """Embed a binary message into integer data using LSB steganography.

    Replaces the least significant bit(s) of cover data with message bits.

    Parameters
    ----------
    cover : array-like of int
        Cover data (e.g., pixel values 0-255).
    message_bits : array-like of int
        Binary message to embed (array of 0s and 1s).
    n_lsb : int
        Number of LSBs to use for embedding (1-4).

    Returns
    -------
    DescriptiveResult
        With ``value`` = stego data (ndarray) and ``extra`` containing
        capacity and PSNR.
    """
    cover = np.asarray(cover, dtype=np.int32).ravel()
    bits = np.asarray(message_bits, dtype=np.int32).ravel()
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("message_bits must be 0 or 1")
    if n_lsb < 1 or n_lsb > 4:
        raise ValueError("n_lsb must be 1-4")

    capacity = len(cover) * n_lsb
    if len(bits) > capacity:
        raise ValueError(f"Message ({len(bits)} bits) exceeds capacity ({capacity} bits)")

    stego = cover.copy()
    mask = (1 << n_lsb) - 1
    clear_mask = ~mask & 0xFF

    bit_idx = 0
    for i in range(len(cover)):
        if bit_idx >= len(bits):
            break
        val = 0
        for b in range(n_lsb):
            if bit_idx < len(bits):
                val |= bits[bit_idx] << b
                bit_idx += 1
        stego[i] = (stego[i] & clear_mask) | val

    mse = float(np.mean((cover.astype(float) - stego.astype(float)) ** 2))
    max_val = float(cover.max()) if cover.max() > 0 else 255.0
    psnr = 10 * np.log10(max_val**2 / max(mse, 1e-30)) if mse > 0 else float("inf")

    return DescriptiveResult(
        name="lsb_embed",
        value=stego,
        extra={"capacity_bits": capacity, "message_bits": len(bits), "n_lsb": n_lsb, "mse": mse, "psnr": float(psnr)},
    )


def lsb_extract(
    stego: np.ndarray,
    n_bits: int,
    *,
    n_lsb: int = 1,
) -> np.ndarray:
    """Extract embedded bits from stego data."""
    stego = np.asarray(stego, dtype=np.int32).ravel()
    bits = []
    for i in range(len(stego)):
        for b in range(n_lsb):
            bits.append((stego[i] >> b) & 1)
            if len(bits) >= n_bits:
                return np.array(bits[:n_bits], dtype=np.int32)
    return np.array(bits[:n_bits], dtype=np.int32)


lsbemb = lsb_embed


def cheatsheet() -> str:
    return 'lsb_embed({}) -> LSB steganography.'
