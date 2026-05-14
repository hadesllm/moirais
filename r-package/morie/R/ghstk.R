# SPDX-License-Identifier: MIT OR Apache-2.0

#' Truncated stick-breaking representation of DP(alpha, G0).
#'
#' @param x Numeric data vector.
#' @param alpha DP concentration parameter (default 1).
#' @param K Integer truncation level (default 50).
#' @param seed Integer RNG seed (default 0).
#' @param base_mean Optional base-measure mean.
#' @param base_sd Optional base-measure sd.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("ghstk", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return Named list with estimate, weights, atoms, effective_K,
#'   trunc_err_bound, n, method.
#' @export
ghosal_stick_breaking_trunc <- function(x, alpha = 1.0, K = 50, seed = 0,
                                          base_mean = NULL, base_sd = NULL,
                                          deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("ghstk", deterministic_seed)
  } else {
    set.seed(seed)
  }
  x <- as.numeric(x); n <- length(x)
  if (is.null(base_mean)) base_mean <- if (n) mean(x) else 0
  if (is.null(base_sd))   base_sd   <- if (n > 1) sd(x) else 1
  base_sd <- max(base_sd, 1e-6)
  V <- stats::rbeta(K, 1, alpha)
  log_cum <- c(0, cumsum(log1p(-V[-K])))
  w <- V * exp(log_cum)
  theta <- stats::rnorm(K, mean = base_mean, sd = base_sd)
  t0 <- if (n) mean(x) else base_mean
  est <- sum(w * (theta <= t0))
  trunc_bound <- (alpha / (alpha + 1))^K
  list(estimate = est, weights = w, atoms = theta,
       effective_K = sum(w > 1e-3),
       trunc_err_bound = trunc_bound, n = n,
       method = "Truncated stick-breaking DP draw (Sethuraman 1994)")
}
