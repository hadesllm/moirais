# SPDX-License-Identifier: AGPL-3.0-or-later

#' k-component Gaussian mixture EM
#'
#' @param y Numeric vector of observations.
#' @param k Integer number of mixture components (default 2).
#' @param maxit Integer maximum EM iterations (default 200).
#' @param tol Numeric convergence tolerance on log-likelihood (default 1e-6).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate (pi, mu, sigma), log_likelihood, n, k, iters, method.
#' @keywords internal
hrzm1 <- function(y, k = 2, maxit = 200, tol = 1e-6, seed = 0) {
  y <- as.numeric(y); n <- length(y)
  if (n < max(10, 3 * k))
    return(list(estimate = NA_real_, n = n,
                method = "mixture-EM (insufficient data)"))
  set.seed(seed)
  mu <- as.numeric(stats::quantile(y, seq(0.1, 0.9, length.out = k)))
  sigma <- rep(stats::sd(y) / k + 1e-3, k)
  pii <- rep(1 / k, k)
  ll_prev <- -Inf; it <- 0
  for (it in 1:maxit) {
    comps <- sapply(1:k, function(j) pii[j] * stats::dnorm(y, mu[j], sigma[j]))
    denom <- rowSums(comps); denom <- ifelse(denom > 0, denom, 1e-12)
    gamma_w <- comps / denom
    Nk <- colSums(gamma_w); Nk <- ifelse(Nk > 0, Nk, 1e-12)
    mu <- colSums(gamma_w * y) / Nk
    sigma <- sqrt(colSums(gamma_w * (y - matrix(mu, n, k, byrow = TRUE))^2) / Nk)
    sigma <- pmax(sigma, 1e-4)
    pii <- Nk / n
    ll <- sum(log(pmax(rowSums(comps), 1e-300)))
    if (abs(ll - ll_prev) < tol) break
    ll_prev <- ll
  }
  list(estimate = list(pi = as.numeric(pii), mu = as.numeric(mu),
                        sigma = as.numeric(sigma)),
       log_likelihood = ll_prev, n = n, k = k, iters = it,
       method = sprintf("%d-component Gaussian mixture EM", k))
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzm1
#' @keywords internal
#' @export
horowitz_mixture_model <- hrzm1
