# SPDX-License-Identifier: MIT OR Apache-2.0

#' Gaussian multiplier bootstrap for Z-estimators
#'
#' G_n_xi(f) = n raised to the power of -1/2 times sum_i xi_i (f(X_i) - P_n f), xi ~ N(0,1).
#'
#' @param x Numeric vector.
#' @param B Number of multiplier replications.
#' @param seed Integer RNG seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("ksr08", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 10.
#' @export
ksr08_kosorok_multiplier_bootstrap <- function(x, B = 1000, seed = 0,
                                               deterministic_seed = NULL) {
  x <- as.numeric(x)
  n <- length(x)
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("ksr08", deterministic_seed)
  } else {
    set.seed(seed)
  }
  pn <- mean(x)
  centred <- x - pn
  xi <- matrix(stats::rnorm(B * n), nrow = B)
  g_xi <- (xi %*% centred) / sqrt(n)
  list(
    estimate = mean(g_xi),
    se       = stats::sd(as.numeric(g_xi)),
    n        = n,
    method   = "Multiplier bootstrap G_n^xi = n^{-1/2} sum xi (f-Pf)"
  )
}

# CANONICAL TEST
# set.seed(0); ksr08_kosorok_multiplier_bootstrap(rnorm(200), B=500, seed=42)

#' @rdname ksr08_kosorok_multiplier_bootstrap
#' @keywords internal
#' @export
kosorok_multiplier_bootstrap <- ksr08_kosorok_multiplier_bootstrap
