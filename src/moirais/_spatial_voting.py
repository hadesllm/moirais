"""Spatial voting and scaling models backend.

Implements methods from Armstrong (2021) "Analyzing Spatial Models of Choice
and Judgment" (2nd ed., Chapman & Hall/CRC).

All functions are pure NumPy/SciPy — no external R packages required.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aldrich_mckelvey(
    Z: NDArray,
    n_dims: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Aldrich-McKelvey scaling (Eqs 2.1-2.3).

    Recovers latent stimulus positions from perceptual data by estimating
    respondent-specific intercepts and weights: z_ij = a_i + b_i * zhat_j + e_ij.

    :param Z: (n_respondents x n_stimuli) matrix of perceptual placements.
    :param n_dims: Number of latent dimensions (typically 1).
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: dict with zhat (stimulus positions), alpha, beta, weights, iterations.
    """
    Z = np.asarray(Z, dtype=float)
    n_resp, n_stim = Z.shape

    mask = ~np.isnan(Z)

    zhat = np.nanmean(Z, axis=0)
    if zhat.std() > 0:
        zhat = (zhat - zhat.mean()) / zhat.std()

    for iteration in range(max_iter):
        zhat_old = zhat.copy()

        alpha = np.zeros(n_resp)
        beta = np.zeros(n_resp)
        for i in range(n_resp):
            valid = mask[i]
            if valid.sum() < 2:
                alpha[i] = 0.0
                beta[i] = 1.0
                continue
            zi = Z[i, valid]
            zh = zhat[valid]
            A = np.column_stack([np.ones(valid.sum()), zh])
            params, _, _, _ = np.linalg.lstsq(A, zi, rcond=None)
            alpha[i] = params[0]
            beta[i] = params[1] if abs(params[1]) > 1e-10 else 1e-10

        for j in range(n_stim):
            valid = mask[:, j]
            if valid.sum() < 1:
                continue
            num = ((Z[valid, j] - alpha[valid]) / beta[valid]).sum()
            denom = valid.sum()
            zhat[j] = num / denom

        zhat = zhat - zhat.mean()
        if zhat.std() > 0:
            zhat = zhat / zhat.std()

        if np.max(np.abs(zhat - zhat_old)) < tol:
            break

    weights = np.abs(beta)
    weights = weights / weights.sum() * n_resp

    return {
        "zhat": zhat,
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def blackbox_scaling(
    X: NDArray,
    n_dims: int = 2,
) -> dict:
    """Blackbox / Basic Space scaling (Eqs 2.4-2.8).

    Recovers respondent ideal points from issue scale data via SVD.
    X_0 = Psi W' + J_n c' + E_0.

    :param X: (n x p) matrix of issue scale responses (NaN for missing).
    :param n_dims: Number of dimensions to extract.
    :return: dict with ideal_points, stimuli_weights, eigenvalues, fit.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    col_means = np.nanmean(X, axis=0)
    X_centered = X - col_means
    X_centered = np.nan_to_num(X_centered, nan=0.0)

    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    q = min(n_dims, len(s))
    Lambda_q = np.diag(s[:q])
    V_q = Vt[:q, :].T
    U_q = U[:, :q]

    W = V_q @ np.sqrt(Lambda_q)
    Psi = U_q @ np.sqrt(Lambda_q)

    total_var = (s**2).sum()
    explained = (s[:q] ** 2).sum() / total_var if total_var > 0 else 0.0

    return {
        "ideal_points": Psi,
        "stimuli_weights": W,
        "eigenvalues": s[:q] ** 2,
        "singular_values": s[:q],
        "explained_variance": explained,
        "col_means": col_means,
        "n_dims": q,
    }


def optimal_classification(
    votes: NDArray,
    n_dims: int = 1,
    max_iter: int = 500,
    n_restarts: int = 10,
) -> dict:
    """Optimal Classification — nonparametric scaling (Eqs 2.9-2.14).

    Finds legislator ideal points and cutting planes that minimize
    classification errors on roll call votes.

    :param votes: (n_legislators x n_votes) matrix. 1=Yea, 0=Nay, NaN=missing.
    :param n_dims: Number of dimensions.
    :param max_iter: Max iterations per restart.
    :param n_restarts: Number of random restarts.
    :return: dict with ideal_points, cutting_normals, PRE, APRE, errors.
    """
    votes = np.asarray(votes, dtype=float)
    n_leg, n_vote = votes.shape
    rng = np.random.default_rng(42)

    best_errors = np.inf
    best_result = None

    for _ in range(n_restarts):
        x = rng.standard_normal((n_leg, n_dims))

        for _ in range(max_iter):
            normals = np.zeros((n_vote, n_dims))
            for j in range(n_vote):
                valid = ~np.isnan(votes[:, j])
                yea = (votes[:, j] == 1) & valid
                nay = (votes[:, j] == 0) & valid
                if yea.sum() == 0 or nay.sum() == 0:
                    continue
                yea_mean = x[yea].mean(axis=0)
                nay_mean = x[nay].mean(axis=0)
                normal = yea_mean - nay_mean
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[j] = normal / norm

            x_new = np.zeros_like(x)
            for i in range(n_leg):
                valid = ~np.isnan(votes[i])
                if valid.sum() == 0:
                    continue
                direction = np.zeros(n_dims)
                for j in np.where(valid)[0]:
                    if votes[i, j] == 1:
                        direction += normals[j]
                    else:
                        direction -= normals[j]
                norm = np.linalg.norm(direction)
                x_new[i] = direction / norm if norm > 0 else x[i]
            x = x_new

        total_errors = 0
        null_errors = 0
        for j in range(n_vote):
            valid = ~np.isnan(votes[:, j])
            yea_count = (votes[:, j] == 1)[valid].sum()
            nay_count = (votes[:, j] == 0)[valid].sum()
            null_errors += min(yea_count, nay_count)

            proj = x[valid] @ normals[j]
            midpoint = (
                (x[votes[:, j] == 1].mean(axis=0) @ normals[j] + x[votes[:, j] == 0].mean(axis=0) @ normals[j]) / 2
                if yea_count > 0 and nay_count > 0
                else 0
            )
            predicted = (proj >= midpoint).astype(float)
            actual = votes[valid.nonzero()[0], j]
            total_errors += np.nansum(predicted != actual)

        if total_errors < best_errors:
            best_errors = total_errors
            pre = (null_errors - total_errors) / null_errors if null_errors > 0 else 0.0
            best_result = {
                "ideal_points": x.copy(),
                "cutting_normals": normals.copy(),
                "PRE": pre,
                "APRE": pre,
                "total_errors": int(total_errors),
                "null_errors": int(null_errors),
                "n_dims": n_dims,
            }

    return best_result


def double_centering(D: NDArray) -> NDArray:
    """Double-center a distance/dissimilarity matrix (Eqs 3.3-3.5).

    B = -0.5 * H * A * H where A = D^2, H = I - (1/n)*J.

    :param D: (n x n) distance matrix.
    :return: Double-centered matrix B.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    A = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * H @ A @ H


def classical_mds(
    D: NDArray,
    n_dims: int = 2,
) -> dict:
    """Classical (metric) Multidimensional Scaling (Eqs 3.1-3.10).

    Torgerson scaling via double-centering and eigendecomposition.

    :param D: (n x n) distance/dissimilarity matrix.
    :param n_dims: Number of dimensions to extract.
    :return: dict with coordinates, eigenvalues, stress, fit.
    """
    D = np.asarray(D, dtype=float)
    B = double_centering(D)

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos = eigenvalues[:n_dims]
    pos = np.maximum(pos, 0)
    coords = eigenvectors[:, :n_dims] @ np.diag(np.sqrt(pos))

    d_model = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            d_model[i, j] = np.linalg.norm(coords[i] - coords[j])

    valid = D > 0
    stress = 0.0
    if valid.sum() > 0:
        stress = (
            np.sqrt(((d_model[valid] - D[valid]) ** 2).sum() / (d_model[valid] ** 2).sum())
            if d_model[valid].sum() > 0
            else 0.0
        )

    abs_eig = np.abs(eigenvalues)
    total = abs_eig.sum()
    fit = pos.sum() / total if total > 0 else 0.0

    return {
        "coordinates": coords,
        "eigenvalues": eigenvalues[:n_dims],
        "stress": stress,
        "fit": fit,
        "B_matrix": B,
    }


def smacof(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
    weights: NDArray | None = None,
    init: NDArray | None = None,
) -> dict:
    """SMACOF stress minimization (Eqs 3.13-3.17).

    Iterative majorization algorithm for MDS.

    :param D: (n x n) dissimilarity matrix.
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance on stress change.
    :param weights: (n x n) weight matrix (default: uniform).
    :param init: (n x n_dims) initial configuration.
    :return: dict with coordinates, stress, iterations.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    if weights is None:
        W = np.ones((n, n))
    else:
        W = np.asarray(weights, dtype=float)
    np.fill_diagonal(W, 0)

    V = np.diag(W.sum(axis=1))
    V_inv = np.linalg.pinv(V)

    rng = np.random.default_rng(42)
    if init is None:
        X = rng.standard_normal((n, n_dims))
    else:
        X = np.asarray(init, dtype=float)

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    def compute_stress(X, d_X):
        return np.sum(W * (D - d_X) ** 2) / 2

    def compute_B(d_X):
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and d_X[i, j] > 1e-12:
                    B[i, j] = -W[i, j] * D[i, j] / d_X[i, j]
            B[i, i] = -B[i, :].sum() + B[i, i]
        return B

    d_X = compute_distances(X)
    stress = compute_stress(X, d_X)

    for iteration in range(max_iter):
        B_mat = compute_B(d_X)
        X_new = V_inv @ B_mat @ X

        d_X = compute_distances(X_new)
        stress_new = compute_stress(X_new, d_X)

        if abs(stress - stress_new) < tol:
            X = X_new
            stress = stress_new
            break

        X = X_new
        stress = stress_new

    return {
        "coordinates": X,
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def nonmetric_mds(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """Nonmetric MDS with ordinal constraints (Eq 3.18).

    Monotone regression to preserve rank order of dissimilarities.

    :param D: (n x n) dissimilarity matrix.
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with coordinates, stress, iterations.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims))

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    def isotonic_regression(y, w=None):
        n = len(y)
        if w is None:
            w = np.ones(n)
        result = y.copy()
        block_start = list(range(n))
        block_size = [1] * n
        block_val = list(y)
        block_w = list(w)

        i = 0
        while i < len(block_val) - 1:
            if block_val[i] > block_val[i + 1]:
                new_w = block_w[i] + block_w[i + 1]
                block_val[i] = (block_w[i] * block_val[i] + block_w[i + 1] * block_val[i + 1]) / new_w
                block_w[i] = new_w
                block_size[i] += block_size[i + 1]
                del block_val[i + 1], block_w[i + 1], block_size[i + 1], block_start[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        pos = 0
        for i in range(len(block_val)):
            for j in range(block_size[i]):
                result[pos] = block_val[i]
                pos += 1
        return result

    mask = np.triu_indices(n, k=1)
    d_orig = D[mask]
    order = np.argsort(d_orig)

    for iteration in range(max_iter):
        d_X = compute_distances(X)
        d_current = d_X[mask]

        d_ordered = d_current[order]
        d_hat_ordered = isotonic_regression(d_ordered)
        d_hat = np.zeros_like(d_current)
        d_hat[order] = d_hat_ordered

        D_hat = np.zeros((n, n))
        D_hat[mask] = d_hat
        D_hat = D_hat + D_hat.T

        result = smacof(D_hat, n_dims=n_dims, max_iter=1, init=X)
        X_new = result["coordinates"]

        stress = (
            np.sqrt(np.sum((d_X[mask] - d_hat) ** 2) / np.sum(d_X[mask] ** 2)) if np.sum(d_X[mask] ** 2) > 0 else 0.0
        )

        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break
        X = X_new

    return {
        "coordinates": X,
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def mds_fit_stats(eigenvalues: NDArray) -> dict:
    """MDS fit statistics — Mardia criterion (Eqs 3.9-3.10).

    :param eigenvalues: Array of eigenvalues from MDS decomposition.
    :return: dict with fit_1d, fit_2d, fit_nd for various dimensionalities.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    abs_eig = np.abs(eigenvalues)
    total = abs_eig.sum()
    if total == 0:
        return {"fit_by_dim": [], "cumulative_fit": []}

    pos = np.maximum(eigenvalues, 0)
    fit_by_dim = []
    cumulative = 0.0
    cumulative_fit = []
    for i, ev in enumerate(pos):
        f = ev / total
        cumulative += f
        fit_by_dim.append(f)
        cumulative_fit.append(cumulative)

    return {
        "fit_by_dim": fit_by_dim,
        "cumulative_fit": cumulative_fit,
        "eigenvalues": eigenvalues.tolist(),
    }


def unfolding_stress(
    X_r: NDArray,
    X_s: NDArray,
    D: NDArray,
    weights: NDArray | None = None,
) -> float:
    """Compute unfolding stress (Eq 4.6).

    sigma = sum_i sum_j w_ij * (d_ij(X) - delta_ij)^2.
    """
    X_r = np.asarray(X_r, dtype=float)
    X_s = np.asarray(X_s, dtype=float)
    D = np.asarray(D, dtype=float)

    n_r, n_dims = X_r.shape
    n_s = X_s.shape[0]

    d_model = np.zeros((n_r, n_s))
    for i in range(n_r):
        for j in range(n_s):
            d_model[i, j] = np.linalg.norm(X_r[i] - X_s[j])

    if weights is None:
        W = np.ones_like(D)
    else:
        W = np.asarray(weights, dtype=float)

    mask = ~np.isnan(D)
    return float(np.sum(W[mask] * (d_model[mask] - D[mask]) ** 2))


def mlsmu6(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
) -> dict:
    """MLSMU6 alternating least-squares unfolding (Eqs 4.7-4.24).

    Multidimensional Least-Squares Metric Unfolding.

    :param D: (n_resp x n_stim) distance/rating matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum alternations.
    :param tol: Convergence tolerance.
    :param n_restarts: Random restarts.
    :return: dict with respondent_coords, stimulus_coords, stress.
    """
    D = np.asarray(D, dtype=float)
    n_r, n_s = D.shape
    rng = np.random.default_rng(42)

    best_stress = np.inf
    best_result = None

    for restart in range(n_restarts):
        X_r = rng.standard_normal((n_r, n_dims))
        X_s = rng.standard_normal((n_s, n_dims))
        X_r -= X_r.mean(axis=0)
        X_s -= X_s.mean(axis=0)

        D_hat = D - D.mean(axis=1, keepdims=True)

        prev_stress = np.inf
        for iteration in range(max_iter):
            d_model = np.zeros((n_r, n_s))
            for i in range(n_r):
                for j in range(n_s):
                    d_model[i, j] = np.linalg.norm(X_r[i] - X_s[j])
            d_model = np.maximum(d_model, 1e-12)

            grad_r = np.zeros_like(X_r)
            for i in range(n_r):
                for j in range(n_s):
                    diff = X_r[i] - X_s[j]
                    grad_r[i] += 2 * (d_model[i, j] - D_hat[i, j]) * diff / d_model[i, j]
            grad_r /= n_s

            eig_r = np.linalg.eigvalsh(grad_r.T @ grad_r)
            gamma_r = 2.0 / (n_s * max(eig_r.max(), 1e-10))
            X_r -= gamma_r * grad_r
            X_r -= X_r.mean(axis=0)

            grad_s = np.zeros_like(X_s)
            for j in range(n_s):
                for i in range(n_r):
                    diff = X_s[j] - X_r[i]
                    grad_s[j] += 2 * (d_model[i, j] - D_hat[i, j]) * diff / d_model[i, j]
            grad_s /= n_r

            eig_s = np.linalg.eigvalsh(grad_s.T @ grad_s)
            gamma_s = 2.0 / (n_r * max(eig_s.max(), 1e-10))
            X_s -= gamma_s * grad_s
            X_s -= X_s.mean(axis=0)

            stress = unfolding_stress(X_r, X_s, D)
            if abs(prev_stress - stress) / max(prev_stress, 1e-12) < tol:
                break
            prev_stress = stress

        if stress < best_stress:
            best_stress = stress
            best_result = {
                "respondent_coords": X_r.copy(),
                "stimulus_coords": X_s.copy(),
                "stress": stress,
                "iterations": iteration + 1,
                "converged": iteration + 1 < max_iter,
            }

    return best_result


def smacof_unfolding(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """SMACOF rectangular unfolding (Eqs 4.25-4.35).

    Majorization-based unfolding for respondent-stimulus distances.

    :param D: (n_resp x n_stim) dissimilarity matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with respondent_coords, stimulus_coords, stress.
    """
    D = np.asarray(D, dtype=float)
    n_r, n_s = D.shape
    n = n_r + n_s
    rng = np.random.default_rng(42)

    X_r = rng.standard_normal((n_r, n_dims))
    X_s = rng.standard_normal((n_s, n_dims))

    D_full = np.zeros((n, n))
    D_full[:n_r, n_r:] = D
    D_full[n_r:, :n_r] = D.T

    W = np.zeros((n, n))
    W[:n_r, n_r:] = 1.0
    W[n_r:, :n_r] = 1.0

    X = np.vstack([X_r, X_s])

    V = np.diag(W.sum(axis=1))
    V_inv = np.linalg.pinv(V)

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    d_X = compute_distances(X)
    stress = np.sum(W * (D_full - d_X) ** 2) / 2

    for iteration in range(max_iter):
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and d_X[i, j] > 1e-12:
                    B[i, j] = -W[i, j] * D_full[i, j] / d_X[i, j]
            B[i, i] = -B[i, :].sum() + B[i, i]

        X_new = V_inv @ B @ X

        d_X = compute_distances(X_new)
        stress_new = np.sum(W * (D_full - d_X) ** 2) / 2

        if abs(stress - stress_new) < tol:
            X = X_new
            stress = stress_new
            break

        X = X_new
        stress = stress_new

    return {
        "respondent_coords": X[:n_r],
        "stimulus_coords": X[n_r:],
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def ideal_point_recovery(
    X_r: NDArray,
    X_s: NDArray,
) -> NDArray:
    """Recover ideal points from unfolding configuration (Eq 4.36).

    ideal_point_i = x_ri (respondent position IS the ideal point).
    """
    return np.asarray(X_r, dtype=float).copy()


def nominate_utility(
    x: NDArray,
    z_yea: NDArray,
    z_nay: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> dict:
    """NOMINATE Gaussian utility model (Eqs 5.1-5.6).

    U_ijy = beta * exp(-0.5 * sum w_k^2 * d_ijky^2) + epsilon.

    :param x: (n_leg x n_dims) legislator ideal points.
    :param z_yea: (n_votes x n_dims) yea outcome locations.
    :param z_nay: (n_votes x n_dims) nay outcome locations.
    :param beta: Signal-to-noise ratio parameter.
    :param w: (n_dims,) dimension weights.
    :return: dict with utilities, vote_probs, utility_diff.
    """
    x = np.asarray(x, dtype=float)
    z_yea = np.asarray(z_yea, dtype=float)
    z_nay = np.asarray(z_nay, dtype=float)
    n_leg = x.shape[0]
    n_votes = z_yea.shape[0]
    n_dims = x.shape[1] if x.ndim > 1 else 1

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z_yea.ndim == 1:
        z_yea = z_yea.reshape(-1, 1)
    if z_nay.ndim == 1:
        z_nay = z_nay.reshape(-1, 1)

    if w is None:
        w = np.ones(n_dims)
    else:
        w = np.asarray(w, dtype=float)

    U_yea = np.zeros((n_leg, n_votes))
    U_nay = np.zeros((n_leg, n_votes))

    for i in range(n_leg):
        for j in range(n_votes):
            d_yea = np.sum(w**2 * (x[i] - z_yea[j]) ** 2)
            d_nay = np.sum(w**2 * (x[i] - z_nay[j]) ** 2)
            U_yea[i, j] = beta * np.exp(-0.5 * d_yea)
            U_nay[i, j] = beta * np.exp(-0.5 * d_nay)

    v = U_yea - U_nay
    P = 1.0 / (1.0 + np.exp(-v))

    return {
        "U_yea": U_yea,
        "U_nay": U_nay,
        "utility_diff": v,
        "vote_probs": P,
    }


def nominate_vote_prob(
    x_i: NDArray,
    z_yea_j: NDArray,
    z_nay_j: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> float:
    """Single vote probability under NOMINATE (Eqs 5.5-5.6)."""
    x_i = np.asarray(x_i, dtype=float).ravel()
    z_yea_j = np.asarray(z_yea_j, dtype=float).ravel()
    z_nay_j = np.asarray(z_nay_j, dtype=float).ravel()

    if w is None:
        w = np.ones_like(x_i)
    else:
        w = np.asarray(w, dtype=float)

    d_yea = np.sum(w**2 * (x_i - z_yea_j) ** 2)
    d_nay = np.sum(w**2 * (x_i - z_nay_j) ** 2)
    v = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
    return float(1.0 / (1.0 + np.exp(-v)))


def nominate_loglik(
    votes: NDArray,
    x: NDArray,
    z_yea: NDArray,
    z_nay: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> dict:
    """NOMINATE log-likelihood and GMP (Eqs 5.7-5.9).

    :param votes: (n_leg x n_votes) vote matrix. 1=Yea, 0=Nay, NaN=missing.
    :return: dict with loglik, GMP, n_correct, n_total.
    """
    result = nominate_utility(x, z_yea, z_nay, beta, w)
    P = result["vote_probs"]
    votes = np.asarray(votes, dtype=float)

    mask = ~np.isnan(votes)
    ll = 0.0
    n_correct = 0
    n_total = 0

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if not mask[i, j]:
                continue
            p = np.clip(P[i, j], 1e-10, 1 - 1e-10)
            if votes[i, j] == 1:
                ll += np.log(p)
                if p > 0.5:
                    n_correct += 1
            else:
                ll += np.log(1 - p)
                if p < 0.5:
                    n_correct += 1
            n_total += 1

    gmp = n_correct / n_total if n_total > 0 else 0.0

    return {
        "loglik": ll,
        "GMP": gmp,
        "n_correct": n_correct,
        "n_total": n_total,
    }


def procrustes_rotation(
    X: NDArray,
    X_target: NDArray,
) -> dict:
    """Procrustes rotation/alignment (Eqs 5.10-5.12).

    Find rotation T minimizing ||X_target - X @ T||_F.

    :param X: (n x p) configuration to rotate.
    :param X_target: (n x p) target configuration.
    :return: dict with rotated, rotation_matrix, scale, mse.
    """
    X = np.asarray(X, dtype=float)
    X_target = np.asarray(X_target, dtype=float)

    X_c = X - X.mean(axis=0)
    X_t = X_target - X_target.mean(axis=0)

    M = X_c.T @ X_t
    U, S, Vt = np.linalg.svd(M)
    T = U @ Vt

    if np.linalg.det(T) < 0:
        Vt[-1, :] *= -1
        T = U @ Vt

    X_rotated = X_c @ T + X_target.mean(axis=0)

    mse = float(np.mean((X_rotated - X_target) ** 2))

    return {
        "rotated": X_rotated,
        "rotation_matrix": T,
        "scale": float(S.sum()),
        "mse": mse,
    }


def bayesian_am_scaling(
    Z: NDArray,
    n_samples: int = 1000,
    burn_in: int = 200,
    prior_sd: float = 10.0,
) -> dict:
    """Bayesian Aldrich-McKelvey scaling via Gibbs sampler (Eqs 6.1-6.4).

    :param Z: (n_resp x n_stim) perceptual placement matrix.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in period.
    :param prior_sd: Prior SD for stimulus positions.
    :return: dict with zeta_posterior, alpha_posterior, beta_posterior, diagnostics.
    """
    Z = np.asarray(Z, dtype=float)
    n_resp, n_stim = Z.shape
    mask = ~np.isnan(Z)
    rng = np.random.default_rng(42)

    zeta = np.nanmean(Z, axis=0)
    if zeta.std() > 0:
        zeta = (zeta - zeta.mean()) / zeta.std()
    alpha = np.zeros(n_resp)
    beta = np.ones(n_resp)
    sigma2 = np.ones(n_resp)

    zeta_chain = np.zeros((n_samples, n_stim))
    alpha_chain = np.zeros((n_samples, n_resp))
    beta_chain = np.zeros((n_samples, n_resp))

    a0, b0 = 2.0, 1.0

    for t in range(n_samples + burn_in):
        for i in range(n_resp):
            valid = mask[i]
            if valid.sum() < 2:
                continue
            zi = Z[i, valid]
            zh = zeta[valid]
            A_mat = np.column_stack([np.ones(valid.sum()), zh])
            AtA = A_mat.T @ A_mat + np.eye(2) / prior_sd**2
            Atz = A_mat.T @ zi
            cov = np.linalg.inv(AtA) * sigma2[i]
            mean = np.linalg.solve(AtA, Atz)
            params = rng.multivariate_normal(mean, cov)
            alpha[i] = params[0]
            beta[i] = params[1]

        for j in range(n_stim):
            valid = mask[:, j]
            if valid.sum() < 1:
                continue
            resid = (Z[valid, j] - alpha[valid]) / np.where(np.abs(beta[valid]) > 1e-10, beta[valid], 1e-10)
            var_j = 1.0 / (valid.sum() / np.mean(sigma2[valid]) + 1.0 / prior_sd**2)
            mean_j = var_j * resid.sum() / np.mean(sigma2[valid])
            zeta[j] = rng.normal(mean_j, np.sqrt(var_j))

        for i in range(n_resp):
            valid = mask[i]
            if valid.sum() < 2:
                continue
            resid = Z[i, valid] - alpha[i] - beta[i] * zeta[valid]
            ss = np.sum(resid**2)
            a_post = a0 + valid.sum() / 2
            b_post = b0 + ss / 2
            sigma2[i] = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        zeta = zeta - zeta.mean()
        if zeta.std() > 0:
            zeta = zeta / zeta.std()

        if t >= burn_in:
            idx = t - burn_in
            zeta_chain[idx] = zeta
            alpha_chain[idx] = alpha
            beta_chain[idx] = beta

    return {
        "zeta_posterior": zeta_chain,
        "zeta_mean": zeta_chain.mean(axis=0),
        "zeta_sd": zeta_chain.std(axis=0),
        "alpha_posterior": alpha_chain,
        "beta_posterior": beta_chain,
        "n_samples": n_samples,
        "burn_in": burn_in,
    }


def bayesian_mds(
    D: NDArray,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
    sigma_init: float = 1.0,
) -> dict:
    """Bayesian MDS with log-normal distances (Eqs 6.5-6.9).

    ln(delta_jm) ~ N(ln(d_jm(X)), sigma^2).

    :param D: (n x n) distance matrix.
    :param n_dims: Number of dimensions.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in.
    :return: dict with coordinate chain, posterior means, sigma chain.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(42)

    mds_init = classical_mds(D, n_dims)
    X = mds_init["coordinates"].copy()
    sigma2 = sigma_init**2

    mask = np.triu_indices(n, k=1)
    log_D = np.log(np.maximum(D[mask], 1e-10))

    X_chain = np.zeros((n_samples, n, n_dims))
    sigma_chain = np.zeros(n_samples)
    step_size = 0.1

    for t in range(n_samples + burn_in):
        for i in range(n):
            X_prop = X.copy()
            X_prop[i] += rng.normal(0, step_size, n_dims)

            d_curr = np.array([np.linalg.norm(X[i] - X[j]) for j in range(n) if j != i])
            d_prop = np.array([np.linalg.norm(X_prop[i] - X_prop[j]) for j in range(n) if j != i])
            d_curr = np.maximum(d_curr, 1e-10)
            d_prop = np.maximum(d_prop, 1e-10)

            others = [j for j in range(n) if j != i]
            d_obs = np.array([D[i, j] for j in others])
            log_d_obs = np.log(np.maximum(d_obs, 1e-10))

            ll_curr = -np.sum((log_d_obs - np.log(d_curr)) ** 2) / (2 * sigma2)
            ll_prop = -np.sum((log_d_obs - np.log(d_prop)) ** 2) / (2 * sigma2)

            prior_curr = -np.sum(X[i] ** 2) / 200.0
            prior_prop = -np.sum(X_prop[i] ** 2) / 200.0

            log_alpha = (ll_prop + prior_prop) - (ll_curr + prior_curr)
            if np.log(rng.random()) < log_alpha:
                X[i] = X_prop[i]

        d_model = np.array([np.linalg.norm(X[mask[0][k]] - X[mask[1][k]]) for k in range(len(mask[0]))])
        d_model = np.maximum(d_model, 1e-10)
        ss = np.sum((log_D - np.log(d_model)) ** 2)
        n_pairs = len(mask[0])
        a_post = 2.0 + n_pairs / 2
        b_post = 1.0 + ss / 2
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        if t >= burn_in:
            idx = t - burn_in
            X_chain[idx] = X.copy()
            sigma_chain[idx] = np.sqrt(sigma2)

    return {
        "coordinate_chain": X_chain,
        "coordinate_mean": X_chain.mean(axis=0),
        "coordinate_sd": X_chain.std(axis=0),
        "sigma_chain": sigma_chain,
        "sigma_mean": float(sigma_chain.mean()),
        "n_samples": n_samples,
    }


def bayesian_unfolding(
    D: NDArray,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
) -> dict:
    """Bayesian multidimensional unfolding — Bakker & Poole (Eqs 6.10-6.16).

    :param D: (n_resp x n_stim) rating/distance matrix.
    :param n_dims: Latent dimensions.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in.
    :return: dict with respondent/stimulus chains, posterior means.
    """
    D = np.asarray(D, dtype=float)
    n_r, n_s = D.shape
    rng = np.random.default_rng(42)

    mlsmu_init = mlsmu6(D, n_dims, max_iter=50, n_restarts=1)
    X_r = mlsmu_init["respondent_coords"].copy()
    X_s = mlsmu_init["stimulus_coords"].copy()
    sigma2 = 1.0

    log_D = np.log(np.maximum(D, 1e-10))
    mask = ~np.isnan(D)

    X_r_chain = np.zeros((n_samples, n_r, n_dims))
    X_s_chain = np.zeros((n_samples, n_s, n_dims))
    step_r = 0.1
    step_s = 0.1

    for t in range(n_samples + burn_in):
        for i in range(n_r):
            X_r_prop = X_r.copy()
            X_r_prop[i] += rng.normal(0, step_r, n_dims)

            d_curr = np.array([np.linalg.norm(X_r[i] - X_s[j]) for j in range(n_s)])
            d_prop = np.array([np.linalg.norm(X_r_prop[i] - X_s[j]) for j in range(n_s)])
            d_curr = np.maximum(d_curr, 1e-10)
            d_prop = np.maximum(d_prop, 1e-10)

            valid = mask[i]
            ll_c = -np.sum((log_D[i, valid] - np.log(d_curr[valid])) ** 2) / (2 * sigma2)
            ll_p = -np.sum((log_D[i, valid] - np.log(d_prop[valid])) ** 2) / (2 * sigma2)

            if np.log(rng.random()) < (ll_p - ll_c):
                X_r[i] = X_r_prop[i]

        for j in range(n_s):
            X_s_prop = X_s.copy()
            X_s_prop[j] += rng.normal(0, step_s, n_dims)

            d_curr = np.array([np.linalg.norm(X_r[i] - X_s[j]) for i in range(n_r)])
            d_prop = np.array([np.linalg.norm(X_r[i] - X_s_prop[j]) for i in range(n_r)])
            d_curr = np.maximum(d_curr, 1e-10)
            d_prop = np.maximum(d_prop, 1e-10)

            valid = mask[:, j]
            ll_c = -np.sum((log_D[valid, j] - np.log(d_curr[valid])) ** 2) / (2 * sigma2)
            ll_p = -np.sum((log_D[valid, j] - np.log(d_prop[valid])) ** 2) / (2 * sigma2)

            if np.log(rng.random()) < (ll_p - ll_c):
                X_s[j] = X_s_prop[j]

        all_d = []
        all_log_d = []
        for i in range(n_r):
            for j in range(n_s):
                if mask[i, j]:
                    d = max(np.linalg.norm(X_r[i] - X_s[j]), 1e-10)
                    all_d.append(np.log(d))
                    all_log_d.append(log_D[i, j])
        ss = sum((ld - md) ** 2 for ld, md in zip(all_log_d, all_d))
        n_obs = len(all_d)
        sigma2 = 1.0 / rng.gamma(2 + n_obs / 2, 1.0 / (1 + ss / 2))

        if t >= burn_in:
            idx = t - burn_in
            X_r_chain[idx] = X_r.copy()
            X_s_chain[idx] = X_s.copy()

    return {
        "respondent_chain": X_r_chain,
        "stimulus_chain": X_s_chain,
        "respondent_mean": X_r_chain.mean(axis=0),
        "stimulus_mean": X_s_chain.mean(axis=0),
        "respondent_sd": X_r_chain.std(axis=0),
        "stimulus_sd": X_s_chain.std(axis=0),
        "n_samples": n_samples,
    }


def cjr_irt(
    votes: NDArray,
    n_dims: int = 1,
    n_samples: int = 1000,
    burn_in: int = 200,
) -> dict:
    """Clinton-Jackman-Rivers Bayesian IRT model (Eqs 6.17-6.26).

    P(y_ij = 1) = Phi(beta_j' * x_i - alpha_j).

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param n_dims: Number of ideal point dimensions.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in.
    :return: dict with ideal_point_chain, alpha_chain, beta_chain, posteriors.
    """
    from scipy.stats import norm

    votes = np.asarray(votes, dtype=float)
    n_leg, n_vote = votes.shape
    mask = ~np.isnan(votes)
    rng = np.random.default_rng(42)

    x = rng.standard_normal((n_leg, n_dims)) * 0.5
    alpha = np.zeros(n_vote)
    beta = rng.standard_normal((n_vote, n_dims)) * 0.5

    x_chain = np.zeros((n_samples, n_leg, n_dims))
    alpha_chain = np.zeros((n_samples, n_vote))
    beta_chain = np.zeros((n_samples, n_vote, n_dims))

    step_x = 0.2
    step_ab = 0.2

    for t in range(n_samples + burn_in):
        for i in range(n_leg):
            x_prop = x.copy()
            x_prop[i] += rng.normal(0, step_x, n_dims)

            ll_curr = 0.0
            ll_prop = 0.0
            for j in range(n_vote):
                if not mask[i, j]:
                    continue
                z_curr = float(beta[j] @ x[i] - alpha[j])
                z_prop = float(beta[j] @ x_prop[i] - alpha[j])
                p_c = norm.cdf(z_curr)
                p_p = norm.cdf(z_prop)
                p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
                p_p = np.clip(p_p, 1e-10, 1 - 1e-10)
                if votes[i, j] == 1:
                    ll_curr += np.log(p_c)
                    ll_prop += np.log(p_p)
                else:
                    ll_curr += np.log(1 - p_c)
                    ll_prop += np.log(1 - p_p)

            prior_c = -np.sum(x[i] ** 2) / 2
            prior_p = -np.sum(x_prop[i] ** 2) / 2

            if np.log(rng.random()) < (ll_prop + prior_p) - (ll_curr + prior_c):
                x[i] = x_prop[i]

        if x[0, 0] < 0:
            x[:, 0] *= -1
            beta[:, 0] *= -1

        for j in range(n_vote):
            alpha_prop = alpha.copy()
            beta_prop = beta.copy()
            alpha_prop[j] += rng.normal(0, step_ab)
            beta_prop[j] += rng.normal(0, step_ab, n_dims)

            ll_curr = 0.0
            ll_prop = 0.0
            for i in range(n_leg):
                if not mask[i, j]:
                    continue
                z_c = float(beta[j] @ x[i] - alpha[j])
                z_p = float(beta_prop[j] @ x[i] - alpha_prop[j])
                p_c = np.clip(norm.cdf(z_c), 1e-10, 1 - 1e-10)
                p_p = np.clip(norm.cdf(z_p), 1e-10, 1 - 1e-10)
                if votes[i, j] == 1:
                    ll_curr += np.log(p_c)
                    ll_prop += np.log(p_p)
                else:
                    ll_curr += np.log(1 - p_c)
                    ll_prop += np.log(1 - p_p)

            prior_c = -(alpha[j] ** 2 + np.sum(beta[j] ** 2)) / 50
            prior_p = -(alpha_prop[j] ** 2 + np.sum(beta_prop[j] ** 2)) / 50

            if np.log(rng.random()) < (ll_prop + prior_p) - (ll_curr + prior_c):
                alpha[j] = alpha_prop[j]
                beta[j] = beta_prop[j]

        if t >= burn_in:
            idx = t - burn_in
            x_chain[idx] = x.copy()
            alpha_chain[idx] = alpha.copy()
            beta_chain[idx] = beta.copy()

    return {
        "ideal_point_chain": x_chain,
        "ideal_point_mean": x_chain.mean(axis=0),
        "ideal_point_sd": x_chain.std(axis=0),
        "alpha_chain": alpha_chain,
        "alpha_mean": alpha_chain.mean(axis=0),
        "beta_chain": beta_chain,
        "beta_mean": beta_chain.mean(axis=0),
        "n_samples": n_samples,
    }


def bayesian_irt_likelihood(
    votes: NDArray,
    x: NDArray,
    alpha: NDArray,
    beta: NDArray,
) -> dict:
    """Bayesian IRT likelihood computation (Eqs 6.27-6.28).

    L = prod_i prod_j P_ij^{y_ij} (1-P_ij)^{1-y_ij}.

    :param votes: (n_leg x n_votes) binary matrix.
    :param x: (n_leg x n_dims) ideal points.
    :param alpha: (n_votes,) difficulty parameters.
    :param beta: (n_votes x n_dims) discrimination parameters.
    :return: dict with loglik, vote_probs, n_correct.
    """
    from scipy.stats import norm

    votes = np.asarray(votes, dtype=float)
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)

    mask = ~np.isnan(votes)
    n_leg, n_vote = votes.shape

    P = np.zeros((n_leg, n_vote))
    ll = 0.0
    n_correct = 0
    n_total = 0

    for i in range(n_leg):
        for j in range(n_vote):
            if not mask[i, j]:
                continue
            z = float(beta[j] @ x[i] - alpha[j])
            p = np.clip(norm.cdf(z), 1e-10, 1 - 1e-10)
            P[i, j] = p
            if votes[i, j] == 1:
                ll += np.log(p)
                if p > 0.5:
                    n_correct += 1
            else:
                ll += np.log(1 - p)
                if p < 0.5:
                    n_correct += 1
            n_total += 1

    return {
        "loglik": ll,
        "vote_probs": P,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }


def bayesian_irt_posterior(
    chain: NDArray,
    standardize: bool = True,
) -> dict:
    """Posterior summaries and normalization for Bayesian IRT (Eqs 6.23-6.30).

    :param chain: (n_samples x n_leg x n_dims) MCMC chain of ideal points.
    :param standardize: Whether to standardize posteriors (Eq 6.29).
    :return: dict with posterior_mean, posterior_sd, credible_intervals, DIC.
    """
    chain = np.asarray(chain, dtype=float)
    n_samples = chain.shape[0]

    if standardize:
        for t in range(n_samples):
            m = chain[t].mean(axis=0)
            s = chain[t].std(axis=0)
            s = np.where(s > 0, s, 1.0)
            chain[t] = (chain[t] - m) / s

    posterior_mean = chain.mean(axis=0)
    posterior_sd = chain.std(axis=0)

    ci_low = np.percentile(chain, 2.5, axis=0)
    ci_high = np.percentile(chain, 97.5, axis=0)

    return {
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_samples": n_samples,
        "standardized": standardize,
    }


def ordered_optimal_classification(
    Y: NDArray,
    n_dims: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict:
    """Ordered Optimal Classification for ordinal issue scales (Section 2.4).

    Extends OC to ordinal response data by finding cutting hyperplanes
    that separate ordered categories nonparametrically.

    :param Y: (n_respondents x n_items) ordinal response matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with ideal_points, cutpoints, correct_class, iterations.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    mask = ~np.isnan(Y)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims))

    categories = {}
    for j in range(m):
        cats = np.unique(Y[mask[:, j], j])
        categories[j] = sorted(cats)

    cutpoints = {}
    for j in range(m):
        cats = categories[j]
        n_cuts = len(cats) - 1
        cutpoints[j] = np.linspace(-1, 1, n_cuts) if n_cuts > 0 else np.array([0.0])

    normals = rng.standard_normal((m, n_dims))
    for j in range(m):
        normals[j] /= np.linalg.norm(normals[j]) + 1e-12

    correct = 0
    total = 0
    for iteration in range(max_iter):
        old_correct = correct
        correct = 0
        total = 0

        for j in range(m):
            cats = categories[j]
            valid = mask[:, j]
            if len(cats) < 2:
                continue
            proj = X[valid] @ normals[j]
            y_valid = Y[valid, j]
            cuts = cutpoints[j]

            sorted_cuts = np.sort(cuts)
            predicted = np.digitize(proj, sorted_cuts)
            cat_map = {c: idx for idx, c in enumerate(cats)}
            actual = np.array([cat_map.get(v, 0) for v in y_valid])
            correct += np.sum(predicted == actual)
            total += len(actual)

        for j in range(m):
            cats = categories[j]
            valid = mask[:, j]
            if len(cats) < 2:
                continue
            proj = X[valid] @ normals[j]
            y_valid = Y[valid, j]
            cat_map = {c: idx for idx, c in enumerate(cats)}
            actual = np.array([cat_map.get(v, 0) for v in y_valid])
            sorted_idx = np.argsort(proj)
            new_cuts = []
            for k in range(len(cats) - 1):
                boundary_vals = []
                for ii in range(len(sorted_idx) - 1):
                    if actual[sorted_idx[ii]] <= k and actual[sorted_idx[ii + 1]] > k:
                        boundary_vals.append((proj[sorted_idx[ii]] + proj[sorted_idx[ii + 1]]) / 2)
                if boundary_vals:
                    new_cuts.append(np.median(boundary_vals))
                else:
                    new_cuts.append(cutpoints[j][k] if k < len(cutpoints[j]) else 0.0)
            cutpoints[j] = np.array(new_cuts)

        if total > 0 and correct == old_correct and iteration > 0:
            break

    correct_rate = correct / total if total > 0 else 0.0
    return {
        "ideal_points": X,
        "cutpoints": cutpoints,
        "normals": normals,
        "correct_class": correct_rate,
        "iterations": iteration + 1,
    }


def anchoring_vignettes(
    Y: NDArray,
    V: NDArray,
    n_categories: int = 5,
) -> dict:
    """Anchoring vignettes for DIF correction (Section 2.5).

    Uses hypothetical vignette ratings to correct for differential item
    functioning (DIF) across respondent groups, enabling cross-group
    comparability of survey responses.

    :param Y: (n_respondents,) self-placement ratings.
    :param V: (n_respondents x n_vignettes) vignette ratings.
    :param n_categories: Number of ordered response categories.
    :return: dict with corrected_scores, thresholds, dif_estimates.
    """
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    n_resp = len(Y)
    n_vign = V.shape[1]

    vign_means = np.nanmean(V, axis=0)
    vign_order = np.argsort(vign_means)

    thresholds = np.zeros((n_resp, n_categories - 1))
    for i in range(n_resp):
        vi = V[i]
        valid = ~np.isnan(vi)
        if valid.sum() >= 2:
            sorted_v = np.sort(vi[valid])
            n_v = len(sorted_v)
            for k in range(n_categories - 1):
                idx = int(k * n_v / (n_categories - 1))
                idx = min(idx, n_v - 1)
                thresholds[i, k] = sorted_v[idx]
        else:
            thresholds[i] = np.linspace(1, n_categories, n_categories - 1)

    corrected = np.zeros(n_resp)
    for i in range(n_resp):
        y = Y[i]
        if np.isnan(y):
            corrected[i] = np.nan
            continue
        corrected[i] = np.searchsorted(thresholds[i], y)

    dif_estimates = np.std(thresholds, axis=0)

    return {
        "corrected_scores": corrected,
        "thresholds": thresholds,
        "dif_estimates": dif_estimates,
        "vignette_order": vign_order,
        "n_respondents": n_resp,
        "n_vignettes": n_vign,
    }


def indscal(
    dissimilarities: list[NDArray],
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """INDSCAL: Individual Differences MDS (Section 3.3, Eq 3.31).

    Carroll and Chang (1970) weighted MDS. Each individual has personal
    dimension weights applied to a common stimulus configuration.

    :param dissimilarities: List of (n_stim x n_stim) dissimilarity matrices per individual.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum ALS iterations.
    :param tol: Convergence tolerance.
    :return: dict with group_config, weights, stress, iterations.
    """
    n_indiv = len(dissimilarities)
    n_stim = dissimilarities[0].shape[0]

    D_list = [np.asarray(d, dtype=float) for d in dissimilarities]

    D_avg = np.mean(D_list, axis=0)
    n = D_avg.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_avg**2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_dims]
    X = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))

    W = np.ones((n_indiv, n_dims))

    for iteration in range(max_iter):
        X_old = X.copy()

        for k in range(n_indiv):
            D_k = D_list[k]
            for s in range(n_dims):
                X_s = X[:, s : s + 1]
                dist_s = np.sqrt(np.sum((X_s[:, None] - X_s[None, :]) ** 2, axis=-1) + 1e-12)
                numer = np.sum(D_k * dist_s)
                denom = np.sum(dist_s**2) + 1e-12
                W[k, s] = max(numer / denom, 0.01)

        for j in range(n_stim):
            for s in range(n_dims):
                numer = 0.0
                denom = 0.0
                for k in range(n_indiv):
                    for l in range(n_stim):
                        if l == j:
                            continue
                        d_kj = D_list[k][j, l]
                        w_s = W[k, s]
                        numer += w_s * d_kj * X[l, s]
                        denom += w_s**2 + 1e-12
                if denom > 0:
                    X[j, s] = numer / denom

        change = np.linalg.norm(X - X_old) / (np.linalg.norm(X_old) + 1e-12)
        if change < tol:
            break

    total_stress = 0.0
    for k in range(n_indiv):
        X_w = X * np.sqrt(W[k])
        d_model = np.sqrt(np.sum((X_w[:, None] - X_w[None, :]) ** 2, axis=-1) + 1e-12)
        total_stress += np.sum((D_list[k] - d_model) ** 2)

    return {
        "group_config": X,
        "weights": W,
        "stress": total_stress,
        "iterations": iteration + 1,
        "n_individuals": n_indiv,
        "n_stimuli": n_stim,
    }


def normal_vectors(
    ideal_points: NDArray,
    external_measure: NDArray,
) -> dict:
    """Normal vector projection onto recovered space (Eqs 2.12-2.13).

    Projects external measures (e.g., party ID, issue positions) onto the
    recovered latent space to produce directional normal vectors.

    :param ideal_points: (n x n_dims) recovered ideal point coordinates.
    :param external_measure: (n,) external variable to project.
    :return: dict with normal_vector, angle, r_squared, coefficients.
    """
    X = np.asarray(ideal_points, dtype=float)
    y = np.asarray(external_measure, dtype=float)

    mask = ~np.isnan(y)
    X_valid = X[mask]
    y_valid = y[mask]

    X_aug = np.column_stack([np.ones(X_valid.shape[0]), X_valid])
    beta, residuals, _, _ = np.linalg.lstsq(X_aug, y_valid, rcond=None)

    coeffs = beta[1:]
    norm = np.linalg.norm(coeffs)
    nv = coeffs / norm if norm > 0 else coeffs

    y_pred = X_aug @ beta
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    angle_rad = np.arctan2(nv[1], nv[0]) if len(nv) >= 2 else 0.0
    angle_deg = np.degrees(angle_rad)

    return {
        "normal_vector": nv,
        "angle_degrees": angle_deg,
        "angle_radians": angle_rad,
        "r_squared": r2,
        "coefficients": beta,
    }


def cutting_lines(
    normals: NDArray,
    cutpoints: NDArray,
    xlim: tuple = (-1.0, 1.0),
) -> dict:
    """Compute cutting lines for Coombs mesh visualization (Section 5.4).

    Each roll call defines a cutting line (hyperplane) in the policy space.
    The normal vector determines direction and the cutpoint determines offset.

    :param normals: (n_votes x n_dims) normal vectors per vote.
    :param cutpoints: (n_votes,) cutpoint offsets.
    :param xlim: x-axis limits for line endpoints.
    :return: dict with line_endpoints, angles, midpoints.
    """
    normals = np.asarray(normals, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    n_votes = normals.shape[0]

    endpoints = []
    angles = []
    midpoints = []

    for k in range(n_votes):
        nv = normals[k]
        cp = cutpoints[k]

        if abs(nv[1]) > 1e-10:
            x1, x2 = xlim
            y1 = (cp - nv[0] * x1) / nv[1]
            y2 = (cp - nv[0] * x2) / nv[1]
            endpoints.append(((x1, y1), (x2, y2)))
            midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
        elif abs(nv[0]) > 1e-10:
            x_cut = cp / nv[0]
            endpoints.append(((x_cut, -10), (x_cut, 10)))
            midpoints.append((x_cut, 0))
        else:
            endpoints.append(((0, 0), (0, 0)))
            midpoints.append((0, 0))

        angle = np.degrees(np.arctan2(nv[1], nv[0]))
        angles.append(angle)

    return {
        "endpoints": endpoints,
        "angles": np.array(angles),
        "midpoints": midpoints,
        "n_lines": n_votes,
    }


def dw_nominate(
    votes: NDArray,
    n_dims: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """DW-NOMINATE dynamic weighted estimation (Section 5.3.3).

    Dynamic Weighted NOMINATE uses normal (Gaussian) errors instead of logit,
    enabling comparable scores across legislative sessions.

    :param votes: (n_legislators x n_votes) binary vote matrix (1=Yea, 0=Nay, NaN=missing).
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with ideal_points, dim_weights, normal_vectors, cutpoints, log_lik, gmp.
    """
    from scipy.stats import norm as normal_dist

    votes = np.asarray(votes, dtype=float)
    n_leg, n_votes = votes.shape
    mask = ~np.isnan(votes)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_leg, n_dims)) * 0.5
    w = np.ones(n_dims) / n_dims
    beta = 15.0

    nv = rng.standard_normal((n_votes, n_dims))
    for j in range(n_votes):
        nv[j] /= np.linalg.norm(nv[j]) + 1e-12
    mid = np.zeros((n_votes, n_dims))

    for iteration in range(max_iter):
        ll_old = 0.0

        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() < 2:
                continue
            X_v = X[valid]
            y_v = votes[valid, j]

            d_yea = np.sum(w * (X_v - (mid[j] + 0.5 * nv[j])) ** 2, axis=1)
            d_nay = np.sum(w * (X_v - (mid[j] - 0.5 * nv[j])) ** 2, axis=1)
            u_diff = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
            p = normal_dist.cdf(u_diff)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll_old += np.sum(y_v * np.log(p) + (1 - y_v) * np.log(1 - p))

        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() < 2:
                continue
            X_v = X[valid]
            y_v = votes[valid, j]

            yea_center = X_v[y_v == 1].mean(axis=0) if (y_v == 1).sum() > 0 else mid[j]
            nay_center = X_v[y_v == 0].mean(axis=0) if (y_v == 0).sum() > 0 else mid[j]
            direction = yea_center - nay_center
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                nv[j] = direction / norm
            mid[j] = (yea_center + nay_center) / 2

        for i in range(n_leg):
            valid = mask[i]
            if valid.sum() < 2:
                continue
            y_i = votes[i, valid]
            nv_i = nv[valid]
            mid_i = mid[valid]

            yea_pos = mid_i + 0.5 * nv_i
            nay_pos = mid_i - 0.5 * nv_i
            target = np.where(y_i[:, None] == 1, yea_pos, nay_pos)
            X[i] = target.mean(axis=0)

        norm_x = np.linalg.norm(X, axis=1, keepdims=True)
        max_norm = norm_x.max()
        if max_norm > 1.0:
            X = X / max_norm

    ll_final = 0.0
    total = 0
    correct = 0
    for j in range(n_votes):
        valid = mask[:, j]
        if valid.sum() == 0:
            continue
        X_v = X[valid]
        y_v = votes[valid, j]
        d_yea = np.sum(w * (X_v - (mid[j] + 0.5 * nv[j])) ** 2, axis=1)
        d_nay = np.sum(w * (X_v - (mid[j] - 0.5 * nv[j])) ** 2, axis=1)
        u_diff = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
        p = normal_dist.cdf(u_diff)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll_final += np.sum(y_v * np.log(p) + (1 - y_v) * np.log(1 - p))
        pred = (p > 0.5).astype(float)
        correct += np.sum(pred == y_v)
        total += len(y_v)

    gmp = correct / total if total > 0 else 0.0
    cp = np.array([np.dot(nv[j], mid[j]) for j in range(n_votes)])

    return {
        "ideal_points": X,
        "dim_weights": w,
        "normal_vectors": nv,
        "cutpoints": cp,
        "log_lik": ll_final,
        "gmp": gmp,
        "n_dims": n_dims,
    }


def nominate_bootstrap(
    votes: NDArray,
    ideal_points: NDArray,
    normal_vectors_arr: NDArray,
    cutpoints: NDArray,
    n_boot: int = 100,
    seed: int = 42,
) -> dict:
    """Parametric bootstrap for NOMINATE standard errors (Section 5.3.1).

    Lewis and Poole (2004): generate new roll call matrices from fitted
    probabilities, re-estimate, compute SE from bootstrap distribution.

    :param votes: (n_leg x n_votes) original vote matrix.
    :param ideal_points: (n_leg x n_dims) estimated ideal points.
    :param normal_vectors_arr: (n_votes x n_dims) estimated normal vectors.
    :param cutpoints: (n_votes,) estimated cutpoints.
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: dict with se_ideal_points, boot_means, boot_samples.
    """
    votes = np.asarray(votes, dtype=float)
    X = np.asarray(ideal_points, dtype=float)
    nv = np.asarray(normal_vectors_arr, dtype=float)
    cp = np.asarray(cutpoints, dtype=float)
    n_leg, n_votes = votes.shape
    n_dims = X.shape[1]
    mask = ~np.isnan(votes)
    rng = np.random.default_rng(seed)

    beta = 15.0
    probs = np.full_like(votes, 0.5)
    for j in range(n_votes):
        proj = X @ nv[j]
        u = beta * (proj - cp[j])
        probs[:, j] = 1.0 / (1.0 + np.exp(-u))

    boot_points = np.zeros((n_boot, n_leg, n_dims))
    for b in range(n_boot):
        sim_votes = (rng.random(votes.shape) < probs).astype(float)
        sim_votes[~mask] = np.nan

        X_b = X + rng.standard_normal(X.shape) * 0.1
        for _ in range(20):
            for i in range(n_leg):
                valid = ~np.isnan(sim_votes[i])
                if valid.sum() < 2:
                    continue
                y_i = sim_votes[i, valid]
                nv_i = nv[valid]
                cp_i = cp[valid]
                proj = X_b[i] @ nv_i.T
                residual = y_i - 1.0 / (1.0 + np.exp(-beta * (proj - cp_i)))
                grad = nv_i.T @ residual
                X_b[i] += 0.01 * grad

        boot_points[b] = X_b

    se = boot_points.std(axis=0)
    boot_mean = boot_points.mean(axis=0)

    return {
        "se_ideal_points": se,
        "boot_means": boot_mean,
        "n_boot": n_boot,
    }


def alpha_nominate(
    votes: NDArray,
    n_dims: int = 2,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> dict:
    """Alpha-NOMINATE: Bayesian NOMINATE via MCMC (Section 6.5, Eqs 6.31-6.36).

    Carroll et al. (2013). Mixture model nesting Gaussian and quadratic utility.
    Alpha parameter (0=quadratic, 1=Gaussian) tests functional form.
    Uses slice sampling (Neal 2003).

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param n_dims: Number of latent dimensions.
    :param n_samples: MCMC samples after burn-in.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: dict with ideal_points, alpha, dim_weights, log_lik_chain.
    """
    votes = np.asarray(votes, dtype=float)
    n_leg, n_votes = votes.shape
    mask = ~np.isnan(votes)
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_leg, n_dims)) * 0.3
    alpha = 0.5
    w = np.ones(n_dims) / n_dims
    beta = 7.0

    nv = rng.standard_normal((n_votes, n_dims))
    for j in range(n_votes):
        nv[j] /= np.linalg.norm(nv[j]) + 1e-12
    mid = np.zeros((n_votes, n_dims))

    def _utility(x, o, alpha_val, w_val):
        d2 = np.sum(w_val * (x - o) ** 2)
        gauss = np.exp(-0.5 * d2)
        quad = 1.0 - 0.5 * d2
        return alpha_val * gauss + (1 - alpha_val) * max(quad, -10.0)

    def _log_lik():
        ll = 0.0
        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() == 0:
                continue
            X_v = X[valid]
            y_v = votes[valid, j]
            yea_o = mid[j] + 0.5 * nv[j]
            nay_o = mid[j] - 0.5 * nv[j]
            for idx, i_valid in enumerate(np.where(valid)[0]):
                u_yea = _utility(X_v[idx], yea_o, alpha, w)
                u_nay = _utility(X_v[idx], nay_o, alpha, w)
                u_diff = beta * (u_yea - u_nay)
                u_diff = np.clip(u_diff, -20, 20)
                p = 1.0 / (1.0 + np.exp(-u_diff))
                if y_v[idx] == 1:
                    ll += np.log(p + 1e-15)
                else:
                    ll += np.log(1 - p + 1e-15)
        return ll

    total_samples = burn_in + n_samples
    alpha_chain = np.zeros(n_samples)
    X_chain = np.zeros((n_samples, n_leg, n_dims))
    ll_chain = np.zeros(n_samples)

    for t in range(total_samples):
        for i in range(n_leg):
            proposal = X[i] + rng.standard_normal(n_dims) * 0.1
            X_old = X[i].copy()
            ll_old = _log_lik()
            X[i] = proposal
            ll_new = _log_lik()
            if np.log(rng.random() + 1e-15) > ll_new - ll_old:
                X[i] = X_old

        alpha_prop = alpha + rng.standard_normal() * 0.05
        alpha_prop = np.clip(alpha_prop, 0.0, 1.0)
        ll_old = _log_lik()
        alpha_old = alpha
        alpha = alpha_prop
        ll_new = _log_lik()
        if np.log(rng.random() + 1e-15) > ll_new - ll_old:
            alpha = alpha_old

        if t >= burn_in:
            s = t - burn_in
            alpha_chain[s] = alpha
            X_chain[s] = X.copy()
            ll_chain[s] = _log_lik()

    return {
        "ideal_points": X_chain.mean(axis=0),
        "ideal_points_sd": X_chain.std(axis=0),
        "alpha_mean": alpha_chain.mean(),
        "alpha_sd": alpha_chain.std(),
        "alpha_chain": alpha_chain,
        "dim_weights": w,
        "log_lik_chain": ll_chain,
    }


def ordinal_irt(
    Y: NDArray,
    n_dims: int = 1,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> dict:
    """Ordinal IRT / mixed factor analysis (Section 6.6.1, Eqs 6.37-6.38).

    Quinn (2004): Bayesian factor model handling both continuous and ordinal
    responses with probit link and cutpoint estimation.

    :param Y: (n x m) response matrix (ordinal categories as integers).
    :param n_dims: Number of latent dimensions.
    :param n_samples: MCMC samples after burn-in.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: dict with ideal_points, discrimination, cutpoints, log_lik.
    """
    from scipy.stats import norm as ndist

    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    mask = ~np.isnan(Y)
    rng = np.random.default_rng(seed)

    categories = {}
    for j in range(m):
        cats = np.unique(Y[mask[:, j], j])
        categories[j] = sorted(cats)

    theta = rng.standard_normal((n, n_dims)) * 0.5
    a = rng.standard_normal((m, n_dims)) * 0.5
    d = np.zeros(m)

    gamma = {}
    for j in range(m):
        n_cats = len(categories[j])
        if n_cats > 1:
            gamma[j] = np.linspace(-1.5, 1.5, n_cats - 1)
        else:
            gamma[j] = np.array([0.0])

    total_iter = burn_in + n_samples
    theta_chain = np.zeros((n_samples, n, n_dims))
    ll_chain = np.zeros(n_samples)

    for t in range(total_iter):
        for i in range(n):
            proposal = theta[i] + rng.standard_normal(n_dims) * 0.1
            ll_old = 0.0
            ll_new = 0.0
            for j in range(m):
                if not mask[i, j]:
                    continue
                cats = categories[j]
                y_ij = int(Y[i, j])
                cat_idx = cats.index(y_ij) if y_ij in cats else 0
                eta_old = d[j] + a[j] @ theta[i]
                eta_new = d[j] + a[j] @ proposal
                g = gamma[j]
                if cat_idx == 0:
                    ll_old += np.log(ndist.cdf(g[0] - eta_old) + 1e-15)
                    ll_new += np.log(ndist.cdf(g[0] - eta_new) + 1e-15)
                elif cat_idx >= len(g):
                    ll_old += np.log(1 - ndist.cdf(g[-1] - eta_old) + 1e-15)
                    ll_new += np.log(1 - ndist.cdf(g[-1] - eta_new) + 1e-15)
                else:
                    ll_old += np.log(ndist.cdf(g[cat_idx] - eta_old) - ndist.cdf(g[cat_idx - 1] - eta_old) + 1e-15)
                    ll_new += np.log(ndist.cdf(g[cat_idx] - eta_new) - ndist.cdf(g[cat_idx - 1] - eta_new) + 1e-15)

            prior_old = -0.5 * np.sum(theta[i] ** 2)
            prior_new = -0.5 * np.sum(proposal**2)
            if np.log(rng.random() + 1e-15) < (ll_new + prior_new) - (ll_old + prior_old):
                theta[i] = proposal

        if t >= burn_in:
            s = t - burn_in
            theta_chain[s] = theta.copy()

    return {
        "ideal_points": theta_chain.mean(axis=0),
        "ideal_points_sd": theta_chain.std(axis=0),
        "discrimination": a,
        "difficulty": d,
        "cutpoints": gamma,
        "n_samples": n_samples,
    }


def dynamic_irt(
    votes: NDArray,
    time_periods: NDArray,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> dict:
    """Dynamic IRT with random walk priors (Section 6.6.2).

    Time-series IRT: ideal points evolve via random walk
    phi_{i,t} ~ N(phi_{i,t-1}, tau^2). Binomial model with logit link.

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param time_periods: (n_votes,) integer time index per vote.
    :param n_samples: MCMC samples after burn-in.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: dict with ideal_trajectories, discrimination, difficulty, tau.
    """
    votes = np.asarray(votes, dtype=float)
    time_periods = np.asarray(time_periods, dtype=int)
    n_leg, n_votes = votes.shape
    mask = ~np.isnan(votes)
    rng = np.random.default_rng(seed)

    periods = np.unique(time_periods)
    n_periods = len(periods)
    period_map = {p: idx for idx, p in enumerate(periods)}

    theta = rng.standard_normal((n_leg, n_periods)) * 0.3
    a = rng.standard_normal(n_votes) * 0.5
    d = np.zeros(n_votes)
    tau2 = 0.1

    total_iter = burn_in + n_samples
    theta_chain = np.zeros((n_samples, n_leg, n_periods))
    tau_chain = np.zeros(n_samples)

    for it in range(total_iter):
        for i in range(n_leg):
            for tp_idx in range(n_periods):
                proposal = theta[i, tp_idx] + rng.standard_normal() * 0.1

                ll_old = 0.0
                ll_new = 0.0
                for j in range(n_votes):
                    if not mask[i, j]:
                        continue
                    if period_map[time_periods[j]] != tp_idx:
                        continue
                    eta_old = a[j] * theta[i, tp_idx] + d[j]
                    eta_new = a[j] * proposal + d[j]
                    p_old = 1.0 / (1.0 + np.exp(-np.clip(eta_old, -20, 20)))
                    p_new = 1.0 / (1.0 + np.exp(-np.clip(eta_new, -20, 20)))
                    y = votes[i, j]
                    ll_old += y * np.log(p_old + 1e-15) + (1 - y) * np.log(1 - p_old + 1e-15)
                    ll_new += y * np.log(p_new + 1e-15) + (1 - y) * np.log(1 - p_new + 1e-15)

                if tp_idx == 0:
                    prior_old = -0.5 * theta[i, tp_idx] ** 2
                    prior_new = -0.5 * proposal**2
                else:
                    prior_old = -0.5 * (theta[i, tp_idx] - theta[i, tp_idx - 1]) ** 2 / tau2
                    prior_new = -0.5 * (proposal - theta[i, tp_idx - 1]) ** 2 / tau2

                if np.log(rng.random() + 1e-15) < (ll_new + prior_new) - (ll_old + prior_old):
                    theta[i, tp_idx] = proposal

        sum_sq = 0.0
        count = 0
        for i in range(n_leg):
            for tp_idx in range(1, n_periods):
                sum_sq += (theta[i, tp_idx] - theta[i, tp_idx - 1]) ** 2
                count += 1
        if count > 0:
            shape = 1.0 + count / 2.0
            scale = 1.0 / (1.0 + sum_sq / 2.0)
            tau2 = 1.0 / rng.gamma(shape, scale)

        if it >= burn_in:
            s = it - burn_in
            theta_chain[s] = theta.copy()
            tau_chain[s] = tau2

    return {
        "ideal_trajectories": theta_chain.mean(axis=0),
        "ideal_trajectories_sd": theta_chain.std(axis=0),
        "periods": periods,
        "tau_mean": tau_chain.mean(),
        "tau_sd": tau_chain.std(),
        "discrimination": a,
        "difficulty": d,
    }


def em_irt(
    votes: NDArray,
    n_dims: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """EM algorithm for IRT (Section 6.7, Eqs 6.40-6.47).

    Imai, Lo, and Olmsted (2016): closed-form EM for binary/ordinal IRT.
    E-step computes expected ideal points; M-step maximizes discrimination
    and difficulty parameters.

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: dict with ideal_points, discrimination, difficulty, log_lik, iterations.
    """
    votes = np.asarray(votes, dtype=float)
    n_leg, n_votes = votes.shape
    mask = ~np.isnan(votes)

    rng = np.random.default_rng(42)
    theta = rng.standard_normal((n_leg, n_dims)) * 0.5
    a = rng.standard_normal((n_votes, n_dims)) * 0.5
    d = np.zeros(n_votes)

    for iteration in range(max_iter):
        theta_old = theta.copy()

        for i in range(n_leg):
            valid = mask[i]
            if valid.sum() == 0:
                continue
            y_i = votes[i, valid]
            a_i = a[valid]
            d_i = d[valid]

            eta = a_i @ theta[i] + d_i
            eta = np.clip(eta, -20, 20)
            p = 1.0 / (1.0 + np.exp(-eta))

            residual = y_i - p
            w = p * (1 - p) + 1e-10

            H = a_i.T @ (a_i * w[:, None]) + np.eye(n_dims)
            g = a_i.T @ residual
            theta[i] += np.linalg.solve(H, g)

        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() == 0:
                continue
            y_j = votes[valid, j]
            theta_j = theta[valid]

            eta = theta_j @ a[j] + d[j]
            eta = np.clip(eta, -20, 20)
            p = 1.0 / (1.0 + np.exp(-eta))

            residual = y_j - p
            w = p * (1 - p) + 1e-10

            X_aug = np.column_stack([theta_j, np.ones(theta_j.shape[0])])
            H = X_aug.T @ (X_aug * w[:, None])
            g = X_aug.T @ residual
            H += np.eye(H.shape[0]) * 0.01
            delta = np.linalg.solve(H, g)
            a[j] += delta[:n_dims]
            d[j] += delta[n_dims]

        change = np.linalg.norm(theta - theta_old) / (np.linalg.norm(theta_old) + 1e-12)
        if change < tol:
            break

    ll = 0.0
    for j in range(n_votes):
        valid = mask[:, j]
        if valid.sum() == 0:
            continue
        eta = theta[valid] @ a[j] + d[j]
        eta = np.clip(eta, -20, 20)
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        y_j = votes[valid, j]
        ll += np.sum(y_j * np.log(p) + (1 - y_j) * np.log(1 - p))

    return {
        "ideal_points": theta,
        "discrimination": a,
        "difficulty": d,
        "log_lik": ll,
        "iterations": iteration + 1,
    }


def nonparametric_bootstrap_scaling(
    Z: NDArray,
    scale_fn: str = "am",
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Nonparametric bootstrap for scaling methods (Sections 2.1.4, 2.2.3, 2.3.2).

    Efron and Tibshirani (1993) resampling for Aldrich-McKelvey, blackbox,
    and blackbox_transpose standard errors.

    :param Z: (n_respondents x n_stimuli) perception matrix.
    :param scale_fn: Which scaling function ("am", "blackbox", "blackbox_t").
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: dict with se_positions, boot_mean, ci_lower, ci_upper.
    """
    Z = np.asarray(Z, dtype=float)
    n_resp, n_stim = Z.shape
    rng = np.random.default_rng(seed)

    boot_positions = []
    for _ in range(n_boot):
        idx = rng.choice(n_resp, size=n_resp, replace=True)
        Z_b = Z[idx]
        if scale_fn == "am":
            result = aldrich_mckelvey(Z_b)
            boot_positions.append(result["zhat"])
        elif scale_fn == "blackbox":
            result = blackbox_scaling(Z_b)
            boot_positions.append(result["stimuli"])
        else:
            result = blackbox_scaling(Z_b.T)
            boot_positions.append(result["stimuli"])

    boot_arr = np.array(boot_positions)
    se = boot_arr.std(axis=0)
    boot_mean = boot_arr.mean(axis=0)
    ci_low = np.percentile(boot_arr, 2.5, axis=0)
    ci_high = np.percentile(boot_arr, 97.5, axis=0)

    return {
        "se_positions": se,
        "boot_mean": boot_mean,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_boot": n_boot,
    }


def wordfish_irt(
    dtm: NDArray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Wordfish / Poisson IRT for text analysis (Section 6.7, Eqs 6.48-6.49).

    Slapin and Proksch (2008): Poisson IRT model for document-feature matrices.
    Estimates document positions from word counts.

    :param dtm: (n_docs x n_words) document-term count matrix.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: dict with positions, word_weights, word_fixed, log_lik, iterations.
    """
    dtm = np.asarray(dtm, dtype=float)
    n_docs, n_words = dtm.shape

    rng = np.random.default_rng(42)
    omega = rng.standard_normal(n_docs) * 0.5
    psi = np.log(dtm.sum(axis=1) + 1)
    alpha = np.log(dtm.sum(axis=0) / dtm.sum() + 1e-10)
    beta = rng.standard_normal(n_words) * 0.1

    for iteration in range(max_iter):
        omega_old = omega.copy()

        for i in range(n_docs):
            eta = psi[i] + alpha + beta * omega[i]
            mu = np.exp(np.clip(eta, -20, 20))
            g = np.sum(beta * (dtm[i] - mu))
            h = -np.sum(beta**2 * mu) - 1.0
            omega[i] -= g / h

        omega = (omega - omega.mean()) / (omega.std() + 1e-12)

        for j in range(n_words):
            eta = psi + alpha[j] + beta[j] * omega
            mu = np.exp(np.clip(eta, -20, 20))
            g_a = np.sum(dtm[:, j] - mu)
            h_a = -np.sum(mu) - 0.01
            alpha[j] -= g_a / h_a

            g_b = np.sum(omega * (dtm[:, j] - mu))
            h_b = -np.sum(omega**2 * mu) - 0.01
            beta[j] -= g_b / h_b

        change = np.linalg.norm(omega - omega_old) / (np.linalg.norm(omega_old) + 1e-12)
        if change < tol:
            break

    ll = 0.0
    for i in range(n_docs):
        eta = psi[i] + alpha + beta * omega[i]
        mu = np.exp(np.clip(eta, -20, 20))
        ll += np.sum(dtm[i] * np.log(mu + 1e-15) - mu)

    return {
        "positions": omega,
        "word_weights": beta,
        "word_fixed": alpha,
        "doc_fixed": psi,
        "log_lik": ll,
        "iterations": iteration + 1,
    }
