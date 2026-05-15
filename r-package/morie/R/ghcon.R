# SPDX-License-Identifier: AGPL-3.0-or-later

#' Schwartz posterior-consistency diagnostic (Bayesian bootstrap).
#'
#' @param x Numeric data vector.
#' @param ref_loc Optional numeric reference location.
#' @param ref_scale Optional numeric reference scale.
#' @param eps Numeric KS-distance tolerance (default 0.1).
#' @param K Integer number of bootstrap draws (default 200).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate, ks_mean, ks_se, schwartz_bound, n, eps, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_posterior_consistency <- function(x, ref_loc = NULL, ref_scale = NULL,
                                           eps = 0.1, K = 200, seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n == 0) return(list(estimate = NA_real_, n = 0,
                          method = "Schwartz consistency (empty)"))
  xs <- sort(x)
  grid <- seq(xs[1] - 1, xs[n] + 1, length.out = 200)
  if (is.null(ref_loc) || is.null(ref_scale)) {
    F_ref <- sapply(grid, function(t) sum(xs <= t)) / n
  } else {
    F_ref <- stats::pnorm(grid, ref_loc, ref_scale)
  }
  ks <- numeric(K)
  for (k in seq_len(K)) {
    if (.gh_have("MCMCpack")) {
      u <- as.numeric(MCMCpack::rdirichlet(1, rep(1, n)))
    } else {
      g <- stats::rgamma(n, shape = 1, rate = 1); u <- g / sum(g)
    }
    cdf <- cumsum(u)  # already in sorted order since xs is sorted
    idx <- findInterval(grid, xs)
    F_draw <- ifelse(idx == 0, 0, cdf[pmin(pmax(idx, 1L), n)])
    ks[k] <- max(abs(F_draw - F_ref))
  }
  list(estimate = mean(ks > eps), ks_mean = mean(ks),
       ks_se = sd(ks) / sqrt(K),
       schwartz_bound = exp(-2 * n * eps^2),
       n = n, eps = eps,
       method = "Schwartz consistency (Bayesian-bootstrap proxy)")
}
