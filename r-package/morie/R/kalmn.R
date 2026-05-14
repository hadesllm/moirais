# SPDX-License-Identifier: MIT OR Apache-2.0

#' Kalman filter predict-update for a linear-Gaussian state-space model
#'
#' Defaults to a univariate local-level model when matrices are omitted.
#'
#' @param x Numeric vector or matrix of observations.
#' @param F Transition matrix (default identity).
#' @param H Observation matrix (default identity).
#' @param Q State-innovation covariance (default sigma^2 I).
#' @param R Observation covariance (default sigma^2 I).
#' @param x0 Initial state mean.
#' @param P0 Initial state covariance.
#' @return Named list with \code{state, state_cov, innovations,
#'   innovation_variance, loglik, n, method}.
#' @export
kalman_filter <- function(x, F = NULL, H = NULL, Q = NULL, R = NULL,
                          x0 = NULL, P0 = NULL) {
  Y <- as.matrix(x); n <- nrow(Y); m <- ncol(Y)
  if (n < 2) stop("Need >=2 obs.")
  if (is.null(F)) F <- diag(m)
  if (is.null(H)) H <- diag(m)
  v0 <- var(diff(Y)) * 0.5
  if (is.null(Q)) Q <- if (is.matrix(v0)) v0 else diag(as.numeric(v0), m)
  if (is.null(R)) R <- if (is.matrix(v0)) v0 else diag(as.numeric(v0), m)
  F <- as.matrix(F); H <- as.matrix(H); Q <- as.matrix(Q); R <- as.matrix(R)
  p <- nrow(F)
  if (is.null(x0)) { x0 <- numeric(p); x0[seq_len(min(p, m))] <- Y[1, seq_len(min(p, m))] }
  if (is.null(P0)) P0 <- diag(1e6, p)
  x_hat <- matrix(0, n, p)
  P_arr <- array(0, c(n, p, p))
  innov <- matrix(0, n, m)
  Sv <- array(0, c(n, m, m))
  xc <- as.numeric(x0); Pc <- P0; ll <- 0
  for (t in seq_len(n)) {
    xp <- F %*% xc
    Pp <- F %*% Pc %*% t(F) + Q
    v <- Y[t, ] - H %*% xp
    S <- H %*% Pp %*% t(H) + R
    Sinv <- tryCatch(solve(S), error = function(e) {
      if (requireNamespace("MASS", quietly = TRUE)) MASS::ginv(S)
      else solve(S + diag(1e-8, nrow(S)))
    })
    K <- Pp %*% t(H) %*% Sinv
    xc <- as.numeric(xp + K %*% v)
    Pc <- (diag(p) - K %*% H) %*% Pp
    x_hat[t, ] <- xc
    P_arr[t, , ] <- Pc
    innov[t, ] <- as.numeric(v)
    Sv[t, , ] <- S
    ld <- determinant(S, logarithm = TRUE)
    if (ld$sign > 0)
      ll <- ll + -0.5 * (m * log(2 * pi) + ld$modulus +
                         sum(v * (Sinv %*% v)))
  }
  list(state = x_hat, state_cov = P_arr,
       innovations = innov, innovation_variance = Sv,
       loglik = as.numeric(ll), n = n,
       method = "Linear Gaussian Kalman filter (base R)")
}
