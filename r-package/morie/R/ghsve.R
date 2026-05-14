# SPDX-License-Identifier: MIT OR Apache-2.0

#' Bernstein-polynomial sieve density estimator (Petrone 1999).
#'
#' @param x Numeric data vector.
#' @param K Optional integer sieve degree (default round(n^(1/3))).
#' @return Named list with estimate, log_lik_per_obs, weights, K, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_sieve_prior <- function(x, K = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 3) return(list(estimate = NA_real_, n = n,
                         method = "Bernstein sieve (n<3)"))
  lo <- min(x) - 1e-6; hi <- max(x) + 1e-6
  u <- (x - lo) / (hi - lo)
  if (is.null(K)) K <- max(2, round(n^(1/3)))
  B <- .gh_bernstein(u, K)
  w <- rep(1/K, K)
  for (it in seq_len(60)) {
    num <- sweep(B, 2, w, "*")
    denom <- pmax(rowSums(num), 1e-12)
    gamma <- num / denom
    w_new <- colMeans(gamma); w_new <- w_new/sum(w_new)
    if (max(abs(w_new - w)) < 1e-8) { w <- w_new; break }
    w <- w_new
  }
  log_lik <- mean(log(pmax(B %*% w, 1e-12)))
  u_bar <- (mean(x) - lo) / (hi - lo)
  B_bar <- .gh_bernstein(u_bar, K)
  f_bar <- as.numeric((B_bar %*% w) / (hi - lo))
  list(estimate = f_bar, log_lik_per_obs = log_lik, weights = w, K = K, n = n,
       method = "Bernstein-polynomial sieve density (Petrone 1999, Ghosal 2001)")
}
