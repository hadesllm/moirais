"""Decomposition backends from Rangayyan & Krishnan Ch. 9.

Implements: matching pursuit, EMD/IMF extraction, NMF,
orthogonal matching pursuit, basis pursuit, and
signal subspace decomposition.
"""

from __future__ import annotations

import numpy as np


def matching_pursuit(
    x: np.ndarray,
    dictionary: np.ndarray | None = None,
    n_atoms: int = 10,
    tolerance: float = 1e-6,
) -> dict:
    n = len(x)
    if dictionary is None:
        dictionary = _gabor_dictionary(n)

    residual = x.copy().astype(float)
    atoms = []
    coefficients = []
    indices = []

    for _ in range(n_atoms):
        correlations = dictionary @ residual
        best_idx = int(np.argmax(np.abs(correlations)))
        coeff = correlations[best_idx]

        if np.abs(coeff) < tolerance:
            break

        atoms.append(dictionary[best_idx])
        coefficients.append(float(coeff))
        indices.append(best_idx)
        residual -= coeff * dictionary[best_idx]

    return {
        "coefficients": np.array(coefficients),
        "indices": np.array(indices, dtype=int),
        "residual": residual,
        "n_atoms_used": len(coefficients),
        "residual_energy": float(np.sum(residual**2)),
    }


def orthogonal_matching_pursuit(
    x: np.ndarray,
    dictionary: np.ndarray | None = None,
    n_atoms: int = 10,
    tolerance: float = 1e-6,
) -> dict:
    n = len(x)
    if dictionary is None:
        dictionary = _gabor_dictionary(n)

    residual = x.copy().astype(float)
    selected = []
    coefficients = []

    for _ in range(n_atoms):
        correlations = dictionary @ residual
        best_idx = int(np.argmax(np.abs(correlations)))

        if np.abs(correlations[best_idx]) < tolerance:
            break

        selected.append(best_idx)
        D_sel = dictionary[selected].T
        coeffs = np.linalg.lstsq(D_sel, x, rcond=None)[0]
        residual = x - D_sel @ coeffs
        coefficients = coeffs.tolist()

    return {
        "coefficients": np.array(coefficients),
        "indices": np.array(selected, dtype=int),
        "residual": residual,
        "n_atoms_used": len(selected),
        "residual_energy": float(np.sum(residual**2)),
    }


def emd(
    x: np.ndarray,
    max_imfs: int = 10,
    max_sifts: int = 100,
    tolerance: float = 0.05,
) -> list[np.ndarray]:
    residual = x.copy().astype(float)
    imfs = []

    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_sifts):
            maxima_idx = _find_extrema(h, kind="max")
            minima_idx = _find_extrema(h, kind="min")

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

            upper = _interpolate_envelope(maxima_idx, h[maxima_idx], len(h))
            lower = _interpolate_envelope(minima_idx, h[minima_idx], len(h))
            mean_env = (upper + lower) / 2

            h_new = h - mean_env
            sd = np.sum((h - h_new) ** 2) / (np.sum(h**2) + 1e-12)
            h = h_new
            if sd < tolerance:
                break

        imfs.append(h)
        residual = residual - h

        if np.sum(residual**2) < 1e-12:
            break

    imfs.append(residual)
    return imfs


def nmf(
    V: np.ndarray,
    n_components: int = 5,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    m, n = V.shape
    rng = np.random.default_rng(42)
    W = np.abs(rng.standard_normal((m, n_components)))
    H = np.abs(rng.standard_normal((n_components, n)))

    for _ in range(max_iter):
        WH = W @ H + 1e-12
        H *= (W.T @ V) / (W.T @ WH + 1e-12)
        WH = W @ H + 1e-12
        W *= (V @ H.T) / (WH @ H.T + 1e-12)
        err = np.linalg.norm(V - W @ H)
        if err < tol:
            break

    return W, H


def _gabor_dictionary(n: int, n_atoms: int = 256) -> np.ndarray:
    rng = np.random.default_rng(0)
    D = np.zeros((n_atoms, n))
    for i in range(n_atoms):
        sigma = rng.uniform(2, n / 4)
        center = rng.uniform(0, n)
        freq = rng.uniform(0, 0.5)
        t = np.arange(n)
        atom = np.exp(-((t - center) ** 2) / (2 * sigma**2)) * np.cos(2 * np.pi * freq * t)
        norm = np.linalg.norm(atom)
        if norm > 0:
            D[i] = atom / norm
    return D


def _find_extrema(x: np.ndarray, kind: str = "max") -> np.ndarray:
    indices = []
    for i in range(1, len(x) - 1):
        if (kind == "max" and x[i] > x[i - 1] and x[i] > x[i + 1]) or (
            kind == "min" and x[i] < x[i - 1] and x[i] < x[i + 1]
        ):
            indices.append(i)
    return np.array(indices, dtype=int)


def _interpolate_envelope(
    indices: np.ndarray,
    values: np.ndarray,
    length: int,
) -> np.ndarray:
    t = np.arange(length)
    return np.interp(t, indices, values)


def ica_decompose(
    X: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, m = X.shape
    if n_components is None:
        n_components = n
    X_centered = X - X.mean(axis=1, keepdims=True)
    cov = np.cov(X_centered)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1][:n_components]
    D = np.diag(1.0 / np.sqrt(evals[idx] + 1e-10))
    X_white = D @ evecs[:, idx].T @ X_centered
    W = np.eye(n_components)
    rng = np.random.default_rng(42)
    W = rng.standard_normal((n_components, n_components))
    W, _ = np.linalg.qr(W)
    for _ in range(100):
        WX = W @ X_white
        g = np.tanh(WX)
        g_prime = 1 - g**2
        W_new = g @ X_white.T / m - np.diag(g_prime.mean(axis=1)) @ W
        W_new, _ = np.linalg.qr(W_new)
        if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < 1e-6:
            W = W_new
            break
        W = W_new
    S = W @ X_white
    mixing = np.linalg.pinv(W @ D @ evecs[:, idx].T)
    return S, mixing


def ensemble_emd(
    x: np.ndarray,
    n_ensembles: int = 100,
    noise_std: float = 0.2,
) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    n = len(x)
    all_imfs = None
    for _ in range(n_ensembles):
        noise = noise_std * np.std(x) * rng.standard_normal(n)
        trial_imfs = emd(x + noise)
        if all_imfs is None:
            all_imfs = [np.zeros(n) for _ in range(len(trial_imfs))]
        for j, imf in enumerate(trial_imfs):
            if j < len(all_imfs):
                all_imfs[j] += imf
    if all_imfs is None:
        return [x.copy()]
    return [imf / n_ensembles for imf in all_imfs]


def dictionary_learning(
    X: np.ndarray,
    n_atoms: int = 10,
    n_iter: int = 50,
    sparsity: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_features, n_samples = X.shape
    D = rng.standard_normal((n_features, n_atoms))
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-10
    codes = np.zeros((n_atoms, n_samples))
    for _ in range(n_iter):
        for j in range(n_samples):
            residual = X[:, j].copy()
            support = []
            for _ in range(sparsity):
                correlations = D.T @ residual
                best = np.argmax(np.abs(correlations))
                if best in support:
                    break
                support.append(best)
                D_sub = D[:, support]
                c, _, _, _ = np.linalg.lstsq(D_sub, X[:, j], rcond=None)
                residual = X[:, j] - D_sub @ c
            codes[:, j] = 0
            for k, s in enumerate(support):
                codes[s, j] = c[k] if k < len(c) else 0
        for k in range(n_atoms):
            usage = np.where(np.abs(codes[k, :]) > 1e-10)[0]
            if len(usage) == 0:
                D[:, k] = rng.standard_normal(n_features)
                D[:, k] /= np.linalg.norm(D[:, k])
                continue
            E = X[:, usage] - D @ codes[:, usage] + np.outer(D[:, k], codes[k, usage])
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            D[:, k] = U[:, 0]
            codes[k, usage] = S[0] * Vt[0, :]
    return D, codes


def mp_time_frequency(
    x: np.ndarray,
    n_atoms: int = 50,
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    n_freq = n // 2
    tfd = np.zeros((n_freq, n))
    residual = x.copy()
    t = np.arange(n) / fs
    f = np.arange(n_freq) * fs / (2 * n_freq)
    for _ in range(n_atoms):
        best_corr = 0
        best_freq = 0
        best_pos = 0
        best_scale = 1
        for scale in [4, 8, 16, 32, 64]:
            if scale > n:
                continue
            for freq_idx in range(0, n_freq, max(1, n_freq // 20)):
                for pos in range(0, n, max(1, n // 20)):
                    half = scale // 2
                    start = max(0, pos - half)
                    end = min(n, pos + half)
                    t_local = np.arange(end - start) / fs
                    atom = np.cos(2 * np.pi * f[freq_idx] * t_local) * np.exp(
                        -((t_local - t_local[len(t_local) // 2]) ** 2) / (2 * (scale / fs) ** 2)
                    )
                    norm = np.linalg.norm(atom)
                    if norm < 1e-10:
                        continue
                    atom /= norm
                    corr = abs(np.dot(residual[start:end], atom))
                    if corr > best_corr:
                        best_corr = corr
                        best_freq = freq_idx
                        best_pos = pos
                        best_scale = scale
        if best_corr < 1e-10:
            break
        half = best_scale // 2
        start = max(0, best_pos - half)
        end = min(n, best_pos + half)
        t_local = np.arange(end - start) / fs
        atom = np.cos(2 * np.pi * f[best_freq] * t_local) * np.exp(
            -((t_local - t_local[len(t_local) // 2]) ** 2) / (2 * (best_scale / fs) ** 2)
        )
        norm = np.linalg.norm(atom)
        if norm > 0:
            atom /= norm
        coeff = np.dot(residual[start:end], atom)
        residual[start:end] -= coeff * atom
        for ti in range(start, end):
            fi_start = max(0, best_freq - 2)
            fi_end = min(n_freq, best_freq + 3)
            tfd[fi_start:fi_end, ti] += coeff**2
    return tfd, t, f


def tfd_features(
    tfd: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
) -> dict:
    total = np.sum(tfd)
    if total == 0:
        return {"mean_time": 0, "mean_freq": 0, "time_spread": 0, "freq_spread": 0, "total_energy": 0}
    p = tfd / total
    marginal_t = np.sum(p, axis=0)
    marginal_f = np.sum(p, axis=1)
    mean_t = float(np.sum(t * marginal_t))
    mean_f = float(np.sum(f * marginal_f))
    var_t = float(np.sum((t - mean_t) ** 2 * marginal_t))
    var_f = float(np.sum((f - mean_f) ** 2 * marginal_f))
    return {
        "mean_time": mean_t,
        "mean_freq": mean_f,
        "time_spread": np.sqrt(var_t),
        "freq_spread": np.sqrt(var_f),
        "total_energy": float(total),
    }


def riemannian_covariance(
    X: np.ndarray,
    metric: str = "riemann",
) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    C = np.cov(X)
    if C.ndim == 0:
        C = np.array([[C]])
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-10)
    if metric == "logeuclid":
        return evecs @ np.diag(np.log(evals)) @ evecs.T
    return C


def ncfs_select(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 5,
) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_feat = X.shape
    scores = np.zeros(n_feat)
    for f in range(n_feat):
        for i in range(n_samples):
            dists = np.abs(X[:, f] - X[i, f])
            weights = np.exp(-dists)
            weights[i] = 0
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            same_class = (y == y[i]).astype(float)
            scores[f] += np.dot(weights, same_class)
    scores /= n_samples
    selected = np.argsort(scores)[::-1][:n_features]
    return selected
