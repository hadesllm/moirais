# SPDX-License-Identifier: AGPL-3.0-or-later
#' RKHS kernel ridge regression (Wahba 1990)
#'
#' Solves min_f  (1/n) ||y - f(x)||^2 + lambda ||f||_H^2 in a Gaussian
#' RKHS.  Closed form alpha = solve(K + n*lambda*I) %*% y.
#'
#' @param x numeric vector or matrix of predictors.
#' @param y numeric outcome vector.
#' @param sigma kernel bandwidth (default: median heuristic).
#' @param lam ridge penalty (default 1e-3).
#' @return list: alpha, fitted, residuals, sigma, lambda, sse, r2, n, method.
#' @keywords internal
rkhsc <- function(x, y, sigma = NULL, lam = 1e-3) {
  if (!is.matrix(x)) x <- matrix(x, ncol = 1)
  y <- as.numeric(y); n <- nrow(x)
  if (n < 2L || length(y) != n)
    return(list(estimate = NA_real_, n = n, method = "RKHS KRR (n<2)"))
  D2 <- as.matrix(stats::dist(x))^2
  if (is.null(sigma)) {
    med <- stats::median(sqrt(D2[D2 > 0]))
    sigma <- if (length(med) && med > 0) med / sqrt(2) else 1
  }
  K <- exp(-D2 / (2 * sigma^2))
  alpha <- solve(K + n * lam * diag(n), y)
  fitted <- as.numeric(K %*% alpha)
  resid <- y - fitted
  sse <- sum(resid^2); sst <- sum((y - mean(y))^2)
  r2 <- if (sst > 0) 1 - sse / sst else NA_real_
  list(alpha = alpha, fitted = fitted, residuals = resid,
       sigma = as.numeric(sigma), lambda = lam,
       sse = sse, r2 = as.numeric(r2),
       estimate = mean(fitted),
       se = sqrt(sse / max(1, n - 1)) / sqrt(n),
       n = as.integer(n),
       method = "RKHS kernel ridge (Wahba 1990)")
}

# CANONICAL TEST
# set.seed(0); x <- seq(0, 1, length.out = 50); y <- sin(2*pi*x) + rnorm(50, sd = 0.05)
# r <- rkhsc(x, y, lam = 1e-4)
# stopifnot(r$r2 > 0.9)

#' @rdname rkhsc
#' @keywords internal
#' @export
rkhs_kernel_regression <- rkhsc
