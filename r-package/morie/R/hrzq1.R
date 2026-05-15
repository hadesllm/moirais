# SPDX-License-Identifier: AGPL-3.0-or-later

#' Koenker-Bassett linear quantile regression
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric response vector.
#' @param tau Numeric quantile in (0, 1) (default 0.5).
#' @return Named list with estimate, se, intercept, tau, n, method.
#' @keywords internal
hrzq1 <- function(x, y, tau = 0.5) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p) || !(tau > 0 && tau < 1))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, tau = tau,
                method = "QReg (insufficient data or invalid tau)"))
  has_int <- isTRUE(all(X[, 1] == 1))
  Xp <- if (!has_int) cbind(1, X) else X
  beta <- .hrz_qreg_irls(Xp, y, tau)
  r <- y - as.numeric(Xp %*% beta)
  h <- (stats::qnorm(1 - 0.05)^(2/3)) *
    ((1.5 * stats::dnorm(stats::qnorm(tau))^2) /
       (2 * stats::qnorm(tau)^2 + 1))^(1/3) * n^(-1/3)
  h <- max(h, 1e-3)
  f0 <- mean(abs(r) < h) / (2 * h); if (f0 < 1e-6) f0 <- 1e-6
  cov_m <- (tau * (1 - tau) / f0^2) * MASS::ginv(t(Xp) %*% Xp)
  se_all <- sqrt(pmax(diag(cov_m), 0))
  if (!has_int) {
    beta_out <- if (ncol(Xp) > 1) beta[-1] else beta
    se_out <- if (ncol(Xp) > 1) se_all[-1] else se_all
    intercept <- as.numeric(beta[1])
  } else { beta_out <- beta; se_out <- se_all; intercept <- NULL }
  list(estimate = if (length(beta_out) == 1) as.numeric(beta_out) else beta_out,
       se = if (length(se_out) == 1) as.numeric(se_out) else se_out,
       intercept = intercept, tau = tau, n = n,
       method = "Koenker-Bassett quantile regression (IRLS)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzq1
#' @keywords internal
#' @export
horowitz_quantile_regression <- hrzq1
