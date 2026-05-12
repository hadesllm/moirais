# SPDX-License-Identifier: GPL-2.0-only
#' Control variates (Nelson 1990)
#'
#' Estimator: theta_cv = mean(Y) - c (mean(C) - mu_c) where c =
#' cov(Y, C)/var(C) is the optimal coefficient.  Variance reduction
#' factor is 1 - rho(Y, C)^2.  The corresponding Python callable
#' (\code{morie.fn.cntrl}) keeps a string-theory central-charge legacy
#' signature; this R parity targets the index-listed control-variates
#' estimator.
#'
#' @param y numeric; outcome samples.
#' @param c_var numeric; correlated control variate, same length as y.
#' @param mu_c numeric; known mean of the control variate.
#' @return list: estimate, se, c_coef, var_ratio_cv_over_crude, n, method.
#' @keywords internal
cntrl_estimator <- function(y, c_var, mu_c) {
  y <- as.numeric(y); cc <- as.numeric(c_var)
  n <- length(y)
  if (n < 2L || length(cc) != n)
    return(list(estimate = NA_real_, n = n, method = "control-variates (bad input)"))
  c_coef <- stats::cov(y, cc) / stats::var(cc)
  theta_cv <- mean(y) - c_coef * (mean(cc) - mu_c)
  se_cv <- sqrt(stats::var(y - c_coef * (cc - mu_c)) / n)
  rho <- stats::cor(y, cc)
  list(estimate = as.numeric(theta_cv),
       se = as.numeric(se_cv),
       c_coef = as.numeric(c_coef),
       var_ratio_cv_over_crude = as.numeric(1 - rho^2),
       n = as.integer(n),
       method = "Control variates (Nelson 1990)")
}

# CANONICAL TEST
# set.seed(0); u <- runif(1000); y <- u + rnorm(1000, sd = 0.01)
# # E[Y] estimated using U as a control variate; mu_c = 0.5
# r <- cntrl_estimator(y, u, 0.5)
# stopifnot(abs(r$estimate - 0.5) < 0.01, r$var_ratio_cv_over_crude < 0.01)

#' @rdname cntrl_estimator
#' @keywords internal
#' @export
control_variates <- cntrl_estimator
