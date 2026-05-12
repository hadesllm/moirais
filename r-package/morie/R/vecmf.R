# SPDX-License-Identifier: GPL-2.0-only

#' Vector error-correction model (VECM)
#'
#' @param Y Numeric matrix (T x k) of I(1) candidate series.
#' @param k_ar Number of lagged differences. Default 1.
#' @param coint_rank Cointegration rank. Default 1.
#' @return Named list with \code{alpha, beta, Gamma, Sigma, loglik, n, k,
#'   rank, method}.
#' @export
vecm <- function(Y, k_ar = 1, coint_rank = 1) {
  Y <- as.matrix(Y); if (nrow(Y) < ncol(Y)) Y <- t(Y)
  Tt <- nrow(Y); k <- ncol(Y)
  if (Tt < 20 || k < 2 || coint_rank < 1 || coint_rank > k)
    stop("Need T>=20, 1<=rank<=k.")
  if (requireNamespace("urca", quietly = TRUE) &&
      requireNamespace("vars", quietly = TRUE)) {
    jres <- urca::ca.jo(Y, type = "trace", ecdet = "none",
                        K = max(k_ar + 1, 2))
    vfit <- vars::vec2var(jres, r = coint_rank)
    return(list(alpha = jres@V[, seq_len(coint_rank), drop = FALSE],
                beta  = jres@V[, seq_len(coint_rank), drop = FALSE],
                Gamma = vfit$A,
                Sigma = summary(jres)$summary,
                loglik = NA_real_,
                n = Tt, k = k, rank = coint_rank,
                method = "VECM via urca::ca.jo + vars::vec2var"))
  }
  dY <- diff(Y); rows <- nrow(dY) - k_ar
  Z0 <- dY[(k_ar + 1):nrow(dY), , drop = FALSE]
  Z1 <- Y[(k_ar + 1):(k_ar + rows), , drop = FALSE]
  Z2 <- if (k_ar == 0) matrix(0, rows, 0)
        else do.call(cbind,
          lapply(seq_len(k_ar), function(i) dY[(k_ar - i + 1):(k_ar - i + rows), ]))
  X <- cbind(Z1, Z2)
  B <- solve(crossprod(X), crossprod(X, Z0))
  Pi_hat <- t(B[seq_len(k), , drop = FALSE])
  sv <- svd(t(Pi_hat))
  alpha <- sv$u[, seq_len(coint_rank), drop = FALSE] *
           rep(sv$d[seq_len(coint_rank)], each = nrow(sv$u))
  beta <- sv$v[, seq_len(coint_rank), drop = FALSE]
  eps <- Z0 - X %*% B
  Sigma <- crossprod(eps) / max(rows - 1, 1)
  list(alpha = alpha, beta = beta, Gamma = list(), Sigma = Sigma,
       loglik = NA_real_, n = Tt, k = k, rank = coint_rank,
       method = "VECM via SVD of OLS Pi (base R)")
}
