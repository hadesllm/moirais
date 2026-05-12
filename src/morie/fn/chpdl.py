# morie.fn -- function file (hadesllm/morie)
"""Chirplet decomposition.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 7.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['chpdl']
def chpdl(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    n_atoms: int = 10,
    chirp_rates: np.ndarray | None = None,
    window_len: int = 64,
) -> DescriptiveResult:
    """Chirplet decomposition via matching pursuit with chirp atoms.

    Decomposes the signal into a sum of Gaussian-windowed chirps
    (chirplets). Each atom is parameterized by center time, center
    frequency, scale, and chirp rate.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    n_atoms : int
        Maximum number of chirplet atoms to extract.
    chirp_rates : array-like or None
        Candidate chirp rates (default linspace).
    window_len : int
        Gaussian window length in samples.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    t = np.arange(n) / fs

    if chirp_rates is None:
        chirp_rates = np.linspace(-fs, fs, 21)
    else:
        chirp_rates = np.asarray(chirp_rates, dtype=float).ravel()

    sigma = window_len / (2.0 * fs)
    residual = x.copy()
    atoms = []

    for _ in range(n_atoms):
        best_coeff = 0.0
        best_params = None
        best_atom = None

        for t0_idx in range(0, n, max(1, window_len // 2)):
            t0 = t[t0_idx]
            g = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
            for cr in chirp_rates:
                for f0 in np.linspace(0, fs / 2, max(2, window_len // 4)):
                    phase = 2 * np.pi * (f0 * (t - t0) + 0.5 * cr * (t - t0) ** 2)
                    atom = g * np.cos(phase)
                    norm = np.sqrt(np.dot(atom, atom) + 1e-20)
                    atom_n = atom / norm
                    coeff = np.dot(residual, atom_n)
                    if abs(coeff) > abs(best_coeff):
                        best_coeff = coeff
                        best_params = {"t0": t0, "f0": f0, "chirp_rate": cr}
                        best_atom = atom_n

        if best_atom is None or abs(best_coeff) < 1e-12:
            break

        atoms.append({**best_params, "coefficient": float(best_coeff)})
        residual -= best_coeff * best_atom

    return DescriptiveResult(
        name="chpdl",
        value=float(len(atoms)),
        extra={
            "atoms": atoms,
            "residual_energy": float(np.dot(residual, residual)),
            "original_energy": float(np.dot(x, x)),
        },
    )


def cheatsheet() -> str:
    return "chpdl({}) -> Chirplet decomposition."
