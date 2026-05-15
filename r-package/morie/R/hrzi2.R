# SPDX-License-Identifier: AGPL-3.0-or-later

#' Density-weighted average derivative
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric response vector.
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @return Named list with estimate, se, bandwidth, n, method.
#' @keywords internal
hrzi2 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(20, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "avg-deriv (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(X[, 1]) else as.numeric(bandwidth)
  if (h <= 0) h <- max(.hrz_silverman(X[, 1]), 1e-6)
  # Pairwise differences
  diffs <- array(0, c(n, n, p))
  for (j in seq_len(p)) diffs[, , j] <- outer(X[, j], X[, j], `-`)
  sq <- apply(diffs^2, c(1, 2), sum) / (h^2)
  K <- exp(-0.5 * sq) / ((2 * pi)^(p/2) * h^p)
  diag(K) <- 0
  grad_f <- matrix(0, n, p)
  for (j in seq_len(p)) grad_f[, j] <- -rowSums(diffs[, , j] * K) / (n * h^2)
  delta <- -(2 / n) * colSums(y * grad_f)
  psi <- -2 * y * grad_f
  if (p == 1) {
    se <- sqrt(stats::var(psi) / n)
  } else {
    se <- sqrt(pmax(diag(stats::cov(psi)) / n, 0))
  }
  list(estimate = if (p == 1) as.numeric(delta) else as.numeric(delta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       bandwidth = h, n = n,
       method = "Powell-Stock-Stoker density-weighted average derivative")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzi2
#' @keywords internal
#' @export
horowitz_average_derivative <- hrzi2
