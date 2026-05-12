# SPDX-License-Identifier: GPL-2.0-only

#' Internal helpers shared across the Horowitz semiparametric suite.
#' Not exported; consumed by hrz* callables only.
#' @keywords internal
#' @name horowitz_helpers
NULL

.hrz_silverman <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 2L) return(1.0)
  s <- stats::sd(x)
  iqr <- diff(stats::quantile(x, c(0.25, 0.75), na.rm = TRUE))
  sigma <- if (iqr > 0) min(s, iqr / 1.349) else s
  if (sigma <= 0) sigma <- max(s, 1e-6)
  unname(1.06 * sigma * n ^ (-1/5))
}


.hrz_R_K_gaussian <- 1.0 / (2.0 * sqrt(pi))


.hrz_gauss_kernel <- function(u) exp(-0.5 * u^2) / sqrt(2 * pi)


.hrz_nw_loo <- function(z, y, h) {
  if (is.null(dim(z))) {
    u <- outer(z, z, `-`) / h
    w <- exp(-0.5 * u^2)
  } else {
    n <- nrow(z); w <- matrix(0, n, n)
    for (j in seq_len(ncol(z))) {
      u <- outer(z[, j], z[, j], `-`) / h
      w <- w + u^2
    }
    w <- exp(-0.5 * w)
  }
  diag(w) <- 0
  wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
  as.numeric((w %*% y) / safe)
}


.hrz_probit_newton <- function(D, Z, maxit = 50, tol = 1e-8) {
  q <- ncol(Z); beta <- rep(0, q)
  for (k in 1:maxit) {
    eta <- pmin(pmax(as.numeric(Z %*% beta), -50), 50)
    p <- stats::pnorm(eta); phi <- stats::dnorm(eta)
    w <- phi * (D - p) / pmax(p * (1 - p), 1e-8)
    Hd <- phi^2 / pmax(p * (1 - p), 1e-8)
    g <- t(Z) %*% w
    H <- t(Z) %*% (Z * Hd)
    step <- tryCatch(solve(H + 1e-8 * diag(q), g),
                     error = function(e) MASS::ginv(H) %*% g)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  as.numeric(beta)
}


.hrz_logit_newton <- function(D, X, maxit = 50, tol = 1e-8) {
  p <- ncol(X); beta <- rep(0, p)
  for (k in 1:maxit) {
    eta <- pmin(pmax(as.numeric(X %*% beta), -50), 50)
    mu <- 1 / (1 + exp(-eta))
    W <- mu * (1 - mu)
    g <- t(X) %*% (D - mu)
    H <- t(X) %*% (X * W)
    step <- tryCatch(solve(H + 1e-8 * diag(p), g),
                     error = function(e) MASS::ginv(H) %*% g)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  1 / (1 + exp(-pmin(pmax(as.numeric(X %*% beta), -50), 50)))
}


.hrz_qreg_irls <- function(X, y, tau = 0.5, maxit = 50, tol = 1e-6) {
  beta <- as.numeric(stats::coef(stats::lm.fit(X, y)))
  for (k in 1:maxit) {
    r <- y - X %*% beta
    w <- ifelse(r > 0, tau / pmax(r, 1e-6),
                (1 - tau) / pmax(-r, 1e-6))
    w <- as.numeric(w)
    new <- tryCatch(solve(t(X) %*% (X * w), t(X) %*% (w * y)),
                    error = function(e) MASS::ginv(t(X) %*% (X * w)) %*% (t(X) %*% (w * y)))
    if (max(abs(new - beta)) < tol) { beta <- new; break }
    beta <- new
  }
  as.numeric(beta)
}


.hrz_hermite <- function(t, J) {
  n <- length(t); H <- matrix(0, n, J)
  H[, 1] <- 1
  if (J > 1) H[, 2] <- t
  if (J > 2) for (k in 3:J) H[, k] <- t * H[, k - 1] - (k - 2) * H[, k - 2]
  for (k in 1:J) H[, k] <- H[, k] / sqrt(max(factorial(k - 1), 1))
  H
}
