# SPDX-License-Identifier: AGPL-3.0-or-later

#' Empirical-Bayes alpha MLE for a DP, given the observed K_n.
#'
#' @param x Numeric data vector.
#' @param alpha_grid Optional numeric grid of alpha values to maximise over.
#' @return Named list with estimate (alpha-hat), K_n, log_lik_at_estimate, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_empirical_bayes <- function(x, alpha_grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, n = n,
                          method = "Empirical Bayes (n<2)"))
  K_n <- length(unique(x))
  if (K_n == n) K_n <- max(2, ceiling(log2(n) + 1))
  neg_ll <- function(a) -(K_n * log(a) + lgamma(a) - lgamma(a + n))
  if (is.null(alpha_grid)) {
    opt <- stats::optimize(neg_ll, interval = c(1e-3, 1e3))
    a_hat <- opt$minimum; ll <- -opt$objective
  } else {
    ll_grid <- -sapply(alpha_grid, neg_ll)
    idx <- which.max(ll_grid); a_hat <- alpha_grid[idx]; ll <- ll_grid[idx]
  }
  list(estimate = a_hat, K_n = K_n, log_lik_at_estimate = ll, n = n,
       method = "Empirical-Bayes alpha for DP (Antoniak 1974 MLE)")
}
