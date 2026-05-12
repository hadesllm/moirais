# SPDX-License-Identifier: GPL-2.0-only

#' Escobar-West augmentation for alpha given K_n with a Gamma(a, b) hyperprior.
#'
#' @param x Numeric data vector.
#' @param a_prior Gamma shape hyperparameter (default 1).
#' @param b_prior Gamma rate hyperparameter (default 1).
#' @param M Integer number of MCMC iterations (default 400).
#' @param seed Integer RNG seed (default 0).
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("ghhbp", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return Named list with estimate (alpha post mean), alpha_se,
#'   alpha_draws, K_n, n, method.
#' @export
ghosal_hierarchical_bayes <- function(x, a_prior = 1.0, b_prior = 1.0,
                                        M = 400, seed = 0,
                                        deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("ghhbp", deterministic_seed)
  } else {
    set.seed(seed)
  }
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, n = n,
                          method = "Hierarchical NP-Bayes (n<2)"))
  K_n <- length(unique(x))
  if (K_n == n) K_n <- max(2, ceiling(log2(n) + 1))
  a <- a_prior; b <- b_prior; alpha <- 1
  draws <- numeric(M)
  for (m in seq_len(M)) {
    eta <- stats::rbeta(1, alpha + 1, n)
    w1 <- a + K_n - 1; w2 <- n * (b - log(eta))
    p_eta <- w1 / (w1 + w2)
    if (stats::runif(1) < p_eta) {
      alpha <- stats::rgamma(1, shape = a + K_n, rate = b - log(eta))
    } else {
      alpha <- stats::rgamma(1, shape = a + K_n - 1, rate = b - log(eta))
    }
    draws[m] <- alpha
  }
  burn <- M %/% 4; chain <- draws[(burn + 1):M]
  list(estimate = mean(chain), alpha_se = sd(chain), alpha_draws = chain,
       K_n = K_n, n = n,
       method = "Escobar-West augmentation for alpha | K_n")
}
