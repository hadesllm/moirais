# SPDX-License-Identifier: AGPL-3.0-or-later

#' Rademacher wild bootstrap for OLS coefficients
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric response vector.
#' @param residuals Optional precomputed residuals.
#' @param B Integer number of bootstrap replications (default 500).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate, se, ci_lower, ci_upper, boot_mean, B, n, method.
#' @keywords internal
hrzw1 <- function(x, y, residuals = NULL, B = 500, seed = 0) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "wild-bootstrap (insufficient data)"))
  beta0 <- as.numeric(MASS::ginv(t(X) %*% X) %*% (t(X) %*% y))
  res <- if (is.null(residuals)) y - as.numeric(X %*% beta0)
         else as.numeric(residuals)
  set.seed(seed)
  XtX_inv <- MASS::ginv(t(X) %*% X)
  boot <- matrix(0, B, p)
  for (b in 1:B) {
    v <- sample(c(-1, 1), n, replace = TRUE)
    y_star <- as.numeric(X %*% beta0) + res * v
    boot[b, ] <- as.numeric(XtX_inv %*% (t(X) %*% y_star))
  }
  mean_b <- colMeans(boot); se <- apply(boot, 2, stats::sd)
  ci_lo <- apply(boot, 2, stats::quantile, 0.025)
  ci_hi <- apply(boot, 2, stats::quantile, 0.975)
  list(estimate = if (p == 1) as.numeric(beta0) else beta0,
       se = if (p == 1) as.numeric(se) else se,
       ci_lower = if (p == 1) as.numeric(ci_lo) else ci_lo,
       ci_upper = if (p == 1) as.numeric(ci_hi) else ci_hi,
       boot_mean = if (p == 1) as.numeric(mean_b) else mean_b,
       B = B, n = n,
       method = "Rademacher wild bootstrap (Mammen 1993)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzw1
#' @keywords internal
#' @export
horowitz_wild_bootstrap <- hrzw1
