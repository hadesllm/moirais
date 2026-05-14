# SPDX-License-Identifier: MIT OR Apache-2.0
#' Antithetic variates (Hammersley & Morton 1956)
#'
#' Crude vs antithetic Monte-Carlo estimator of E_U of f(U); reports the
#' variance-reduction ratio var_AV / var_crude.
#'
#' @param x optional U(0,1) sample; if NULL, draw N points.
#' @param f integrand on (0, 1).  Default f(u) = u.
#' @param N integer (used when x is NULL).
#' @param seed integer.
#' @return list: estimate, estimate_crude, se, var_ratio_av_over_crude,
#'   n_pairs, method.
#' @keywords internal
antth <- function(x = NULL, f = NULL, N = 1000L, seed = 42L) {
  if (is.null(f)) f <- function(u) u
  set.seed(seed)
  if (is.null(x)) u <- stats::runif(N) else {
    u <- as.numeric(x)
    if (min(u) < 0 || max(u) > 1)
      u <- (rank(u)) / (length(u) + 1)
  }
  n <- length(u)
  fu <- vapply(u, f, numeric(1))
  fu_anti <- vapply(1 - u, f, numeric(1))
  paired <- 0.5 * (fu + fu_anti)
  est_av <- mean(paired)
  se_av  <- stats::sd(paired) / sqrt(n)
  est_crude <- mean(fu)
  var_crude <- stats::var(fu) / n
  ratio <- if (var_crude > 0) se_av^2 / var_crude else NA_real_
  list(estimate = as.numeric(est_av),
       estimate_crude = as.numeric(est_crude),
       se = as.numeric(se_av),
       var_ratio_av_over_crude = as.numeric(ratio),
       n_pairs = as.integer(n),
       method = "Antithetic variates (Hammersley & Morton 1956)")
}

# CANONICAL TEST
# r <- antth(N = 2000, seed = 0)
# stopifnot(abs(r$estimate - 0.5) < 0.05)
# stopifnot(r$se < 1e-9)   # antithetic var for monotone f(u)=u is 0

#' @rdname antth
#' @keywords internal
#' @export
antithetic_variates <- antth
