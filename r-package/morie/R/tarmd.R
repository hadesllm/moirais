# SPDX-License-Identifier: GPL-2.0-only

#' Two-regime self-exciting threshold autoregressive (SETAR) model
#'
#' @param x Numeric univariate series.
#' @param p AR order in each regime. Default 1.
#' @param d Delay parameter for the threshold variable. Default 1.
#' @param n_grid Grid size for threshold search. Default 50.
#' @return Named list with \code{threshold, phi_lower, phi_upper, p, d,
#'   regime_sizes, sse, n, method}.
#' @export
threshold_autoregression <- function(x, p = 1, d = 1, n_grid = 50) {
  y <- as.numeric(x); n <- length(y); start <- max(p, d)
  if (n - start < 4 * (p + 1))
    stop("Series too short for SETAR(p, d).")
  Y <- y[(start + 1):n]
  X <- cbind(1, do.call(cbind,
    lapply(seq_len(p), function(i) y[(start - i + 1):(n - i)])))
  Z <- y[(start - d + 1):(n - d)]
  ql <- as.numeric(quantile(Z, 0.15))
  qh <- as.numeric(quantile(Z, 0.85))
  grid <- seq(ql, qh, length.out = n_grid)
  best <- list(sse = Inf, threshold = NA, phi_lo = NULL, phi_hi = NULL,
               sizes = NULL)
  for (c in grid) {
    lo <- Z <= c; hi <- !lo
    if (sum(lo) < 2 * (p + 1) || sum(hi) < 2 * (p + 1)) next
    phi_lo <- lsfit(X[lo, , drop = FALSE], Y[lo], intercept = FALSE)$coef
    phi_hi <- lsfit(X[hi, , drop = FALSE], Y[hi], intercept = FALSE)$coef
    sse <- sum((Y[lo] - X[lo, , drop = FALSE] %*% phi_lo)^2) +
           sum((Y[hi] - X[hi, , drop = FALSE] %*% phi_hi)^2)
    if (sse < best$sse) {
      best <- list(sse = sse, threshold = c,
                   phi_lo = phi_lo, phi_hi = phi_hi,
                   sizes = c(lower = sum(lo), upper = sum(hi)))
    }
  }
  if (is.null(best$phi_lo))
    stop("Could not find admissible threshold grid point.")
  list(threshold = best$threshold,
       phi_lower = best$phi_lo, phi_upper = best$phi_hi,
       p = p, d = d, regime_sizes = best$sizes,
       sse = best$sse, n = n,
       method = sprintf("SETAR(p=%d, d=%d) via grid-search OLS", p, d))
}
