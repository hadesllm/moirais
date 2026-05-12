# SPDX-License-Identifier: GPL-2.0-only

#' GP nonparametric regression
#'
#' Wraps \code{ghosal_gp_squared_exponential}.
#'
#' @param x Numeric vector or matrix of input points.
#' @param y Numeric response vector.
#' @param length_scale Optional kernel length-scale.
#' @param sigma_f Numeric signal sd (default 1).
#' @param noise Optional observation noise sd.
#' @return Named list with estimate, se, mu, sd, ci_lower, ci_upper, r2,
#'   log_marginal, length_scale, noise, n, method.
#' @export
ghosal_np_regression <- function(x, y, length_scale = NULL,
                                   sigma_f = 1.0, noise = NULL) {
  gp <- ghosal_gp_squared_exponential(x, y, length_scale = length_scale,
                                       sigma_f = sigma_f, noise = noise)
  yv <- as.numeric(y); mu <- gp$mu; sd_ <- gp$sd
  ss_tot <- sum((yv - mean(yv))^2); ss_res <- sum((yv - mu)^2)
  r2 <- 1 - ss_res / max(ss_tot, 1e-12)
  var_pred <- sd_^2 + gp$noise^2
  log_marg <- -0.5 * sum((yv - mu)^2 / var_pred + log(2*pi*var_pred))
  list(estimate = mean(mu), se = mean(sd_), mu = mu, sd = sd_,
       ci_lower = mu - 1.96 * sqrt(var_pred),
       ci_upper = mu + 1.96 * sqrt(var_pred),
       r2 = r2, log_marginal = log_marg,
       length_scale = gp$length_scale, noise = gp$noise, n = length(yv),
       method = "GP regression posterior")
}
