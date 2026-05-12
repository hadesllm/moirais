# SPDX-License-Identifier: GPL-2.0-only

#' Bootstrap consistency for the empirical process
#'
#' G_n^*(f) = sqrt(n)(P_n^* - P_n)(f).  Monte-Carlo bootstrap of the
#' sample mean; returns mean/SD of G_n^* across B replications.
#'
#' @param x Numeric vector.
#' @param B Number of bootstrap replications (default 1000).
#' @param seed Integer RNG seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("ksr07", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 10.
#' @export
ksr07_kosorok_bootstrap_empirical <- function(x, B = 1000, seed = 0,
                                              deterministic_seed = NULL) {
  x <- as.numeric(x)
  n <- length(x)
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("ksr07", deterministic_seed)
  } else {
    set.seed(seed)
  }
  pn <- mean(x)
  boot_means <- replicate(B, mean(sample(x, n, replace = TRUE)))
  gn_star <- sqrt(n) * (boot_means - pn)
  list(
    estimate = mean(gn_star),
    se       = stats::sd(gn_star),
    n        = n,
    method   = "Bootstrap G_n^*(f) = sqrt(n)(P_n^* - P_n)"
  )
}

# CANONICAL TEST
# set.seed(0); ksr07_kosorok_bootstrap_empirical(rnorm(200), B=500, seed=42)

#' @rdname ksr07_kosorok_bootstrap_empirical
#' @keywords internal
#' @export
kosorok_bootstrap_empirical <- ksr07_kosorok_bootstrap_empirical
