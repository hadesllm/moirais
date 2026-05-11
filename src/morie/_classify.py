"""Classification backends from Rangayyan & Krishnan Ch. 10.

Implements: Fisher LDA, Mahalanobis classifier, Bayes classifier, QDA,
logistic classifier, RBF network, confusion matrix, k-fold CV, LOOCV,
Bhattacharyya divergence, 1D CNN, LSTM for biosignals.
"""

from __future__ import annotations

import numpy as np


def fisher_lda(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    classes = np.unique(y)
    if len(classes) != 2:
        c0 = classes[0]
        c1 = classes[1] if len(classes) > 1 else classes[0]
    else:
        c0, c1 = classes
    X0 = X[y == c0]
    X1 = X[y == c1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    S0 = np.cov(X0.T) if X0.shape[0] > 1 else np.eye(X.shape[1]) * 1e-6
    S1 = np.cov(X1.T) if X1.shape[0] > 1 else np.eye(X.shape[1]) * 1e-6
    if S0.ndim == 0:
        S0 = np.array([[S0]])
    if S1.ndim == 0:
        S1 = np.array([[S1]])
    Sw = S0 + S1
    try:
        w = np.linalg.solve(Sw, mu1 - mu0)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Sw, mu1 - mu0, rcond=None)[0]
    w = w / (np.linalg.norm(w) + 1e-10)
    threshold = float(w @ (mu0 + mu1) / 2)
    return w, np.array([mu0, mu1]), threshold


def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> float:
    diff = x - mu
    if cov.ndim == 0:
        cov = np.array([[cov]])
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    return float(np.sqrt(diff @ cov_inv @ diff))


def bayes_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_test = X_test.shape[0] if X_test.ndim > 1 else 1
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    posteriors = np.zeros((n_test, n_classes))
    for i, c in enumerate(classes):
        Xc = X_train[y_train == c]
        prior = len(Xc) / len(y_train)
        mu = Xc.mean(axis=0)
        cov = np.cov(Xc.T) if Xc.shape[0] > 1 else np.eye(X_train.shape[1]) * 1e-6
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(cov.shape[0]) * 1e-6
        d = X_train.shape[1]
        det = np.linalg.det(cov)
        if det <= 0:
            det = 1e-10
        cov_inv = np.linalg.pinv(cov)
        for j in range(n_test):
            diff = X_test[j] - mu
            exponent = -0.5 * diff @ cov_inv @ diff
            posteriors[j, i] = np.log(prior) + exponent - 0.5 * np.log(det) - d / 2 * np.log(2 * np.pi)
    predictions = classes[np.argmax(posteriors, axis=1)]
    return predictions, posteriors


def qda_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    predictions, _ = bayes_classifier(X_train, y_train, X_test)
    return predictions


def logistic_classify(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    n_iter: int = 1000,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1)
    for _ in range(n_iter):
        z = X_aug @ w
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        grad = X_aug.T @ (p - y) / n
        w -= lr * grad
    z = X_aug @ w
    z = np.clip(z, -500, 500)
    p = 1 / (1 + np.exp(-z))
    preds = (p >= 0.5).astype(float)
    accuracy = float(np.mean(preds == y))
    return w, accuracy


def rbf_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_centers: int = 10,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(n_centers, len(X_train)), replace=False)
    centers = X_train[idx]
    n = len(X_train)
    Phi = np.zeros((n, n_centers))
    for i in range(n):
        for j in range(n_centers):
            dist = np.linalg.norm(X_train[i] - centers[j])
            Phi[i, j] = np.exp(-(dist**2) / (2 * sigma**2))
    weights, _, _, _ = np.linalg.lstsq(Phi, y_train, rcond=None)
    return weights, centers, np.array([sigma])


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0
    if n == 2:
        tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    return {"confusion_matrix": cm, "accuracy": accuracy, "classes": classes.tolist()}


def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> tuple[float, np.ndarray]:
    n = len(y)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    accuracies = np.zeros(k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        preds, _ = bayes_classifier(X_train, y_train, X_test)
        accuracies[i] = float(np.mean(preds == y_test))
    return float(np.mean(accuracies)), accuracies


def loocv(
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    n = len(y)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred, _ = bayes_classifier(X[mask], y[mask], X[i : i + 1])
        if pred[0] == y[i]:
            correct += 1
    return float(correct / n)


def bhattacharyya_divergence(
    X1: np.ndarray,
    X2: np.ndarray,
) -> float:
    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    S1 = np.cov(X1.T) if X1.shape[0] > 1 else np.eye(X1.shape[1]) * 1e-6
    S2 = np.cov(X2.T) if X2.shape[0] > 1 else np.eye(X2.shape[1]) * 1e-6
    if S1.ndim == 0:
        S1 = np.array([[S1]])
    if S2.ndim == 0:
        S2 = np.array([[S2]])
    S = (S1 + S2) / 2
    diff = mu1 - mu2
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    term1 = 0.125 * diff @ S_inv @ diff
    det1 = max(np.linalg.det(S1), 1e-300)
    det2 = max(np.linalg.det(S2), 1e-300)
    det_s = max(np.linalg.det(S), 1e-300)
    term2 = 0.5 * np.log(det_s / np.sqrt(det1 * det2))
    return float(term1 + term2)


def cnn_biosignal(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 10,
    n_filters: int = 8,
    kernel_size: int = 5,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(42)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    filters = rng.standard_normal((n_filters, kernel_size)) * 0.1
    n_conv_out = n_features - kernel_size + 1
    if n_conv_out < 1:
        n_conv_out = 1
    n_pool = max(1, n_conv_out // 2)
    n_classes = len(np.unique(y))
    W_fc = rng.standard_normal((n_filters * n_pool, n_classes)) * 0.1
    b_fc = np.zeros(n_classes)
    lr = 0.01
    for _ in range(n_epochs):
        for i in range(n_samples):
            conv_out = np.zeros((n_filters, max(n_conv_out, 1)))
            for f in range(n_filters):
                for j in range(n_conv_out):
                    if j + kernel_size <= n_features:
                        conv_out[f, j] = max(0, np.dot(X[i, j : j + kernel_size], filters[f]))
            pooled = np.zeros((n_filters, n_pool))
            for f in range(n_filters):
                for j in range(n_pool):
                    start = j * 2
                    end = min(start + 2, n_conv_out)
                    pooled[f, j] = np.max(conv_out[f, start:end])
            flat = pooled.flatten()
            if len(flat) != W_fc.shape[0]:
                continue
            logits = flat @ W_fc + b_fc
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            target = np.zeros(n_classes)
            target_idx = int(y[i]) if n_classes > 2 else int(y[i])
            if target_idx < n_classes:
                target[target_idx] = 1
            grad = probs - target
            W_fc -= lr * np.outer(flat, grad)
            b_fc -= lr * grad
    correct = 0
    for i in range(n_samples):
        conv_out = np.zeros((n_filters, max(n_conv_out, 1)))
        for f in range(n_filters):
            for j in range(n_conv_out):
                if j + kernel_size <= n_features:
                    conv_out[f, j] = max(0, np.dot(X[i, j : j + kernel_size], filters[f]))
        pooled = np.zeros((n_filters, n_pool))
        for f in range(n_filters):
            for j in range(n_pool):
                start = j * 2
                end = min(start + 2, n_conv_out)
                pooled[f, j] = np.max(conv_out[f, start:end])
        flat = pooled.flatten()
        if len(flat) == W_fc.shape[0]:
            logits = flat @ W_fc + b_fc
            if np.argmax(logits) == int(y[i]):
                correct += 1
    accuracy = correct / n_samples if n_samples > 0 else 0
    return filters, float(accuracy)


def lstm_biosignal(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 10,
    hidden_size: int = 16,
) -> tuple[dict, float]:
    rng = np.random.default_rng(42)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, seq_len = X.shape
    n_classes = len(np.unique(y))
    scale = 0.1
    Wf = rng.standard_normal((hidden_size, hidden_size + 1)) * scale
    Wi = rng.standard_normal((hidden_size, hidden_size + 1)) * scale
    Wc = rng.standard_normal((hidden_size, hidden_size + 1)) * scale
    Wo = rng.standard_normal((hidden_size, hidden_size + 1)) * scale
    W_out = rng.standard_normal((hidden_size, n_classes)) * scale

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    correct = 0
    for i in range(n_samples):
        h = np.zeros(hidden_size)
        c = np.zeros(hidden_size)
        for t in range(seq_len):
            x_t = np.array([X[i, t]])
            combined = np.concatenate([h, x_t])
            f_gate = sigmoid(Wf @ combined)
            i_gate = sigmoid(Wi @ combined)
            c_cand = np.tanh(Wc @ combined)
            c = f_gate * c + i_gate * c_cand
            o_gate = sigmoid(Wo @ combined)
            h = o_gate * np.tanh(c)
        logits = h @ W_out
        if np.argmax(logits) == int(y[i]):
            correct += 1
    accuracy = correct / n_samples if n_samples > 0 else 0
    params = {"Wf": Wf, "Wi": Wi, "Wc": Wc, "Wo": Wo, "W_out": W_out, "hidden_size": hidden_size}
    return params, float(accuracy)
