# SPDX-License-Identifier: MIT OR Apache-2.0

#' Dirichlet-process posterior (conjugate update)
#'
#' Posterior of G given X_1, ..., X_n for G ~ DP(alpha, G0) with
#' G0 = N(base_mean, base_sd^2).  Returns the posterior-mean CDF
#' evaluated on a grid plus the headline `estimate` at `mean(x)`.
#' @param x numeric vector.
#' @param alpha concentration.
#' @param base_mean,base_sd base measure (N).
#' @param grid optional grid (default: 51 pts spanning x).
#' @return named list with `estimate`, `cdf_grid`, `cdf_post`,
#'   `cdf_var`, `alpha_post`, `n`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_dirichlet_posterior <- function(x, alpha = 1.0, base_mean = 0,
                                         base_sd = 1, grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (is.null(grid)) {
    if (n == 0) grid <- seq(base_mean - 3*base_sd, base_mean + 3*base_sd, length.out = 51)
    else {
      pad <- max(1e-6, 0.1 * (max(x) - min(x) + 1))
      grid <- seq(min(x) - pad, max(x) + pad, length.out = 51)
    }
  }
  alpha_post <- alpha + n
  G0_t <- stats::pnorm(grid, mean = base_mean, sd = base_sd)
  emp_t <- if (n > 0) sapply(grid, function(t) sum(x <= t)) else rep(0, length(grid))
  F_post <- (alpha * G0_t + emp_t) / alpha_post
  var_post <- F_post * (1 - F_post) / (alpha_post + 1)
  t0 <- if (n > 0) mean(x) else base_mean
  G0_t0 <- stats::pnorm(t0, mean = base_mean, sd = base_sd)
  emp_t0 <- if (n > 0) sum(x <= t0) else 0
  est <- (alpha * G0_t0 + emp_t0) / alpha_post
  list(estimate = est, alpha_post = alpha_post, n = n,
       cdf_grid = grid, cdf_post = F_post, cdf_var = var_post,
       method = "Dirichlet process posterior (conjugate)")
}
