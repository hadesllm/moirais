# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: MISE decomposition for Gaussian KDE (Ch 1)
#'
#' \deqn{\mathrm{MISE}(h) = (h^4/4)\mu_2^2 R(f'') + R(K)/(nh).}
#' With normal-reference plug-in \eqn{R(f'')=3/(8\sqrt\pi \sigma^5)}.
#' Returns h_opt minimising MISE.
#'
#' @param x Numeric vector.
#' @param h Bandwidth (default = Silverman).
#' @return Named list with estimate (MISE), bias_part, var_part, h, h_opt,
#'   R_fpp, sigma, n, method.
#' @importFrom stats sd
#' @export
fzmis <- function(x, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(estimate = NA_real_, n = n,
                           method = "fzmis - too few obs"))
  sigma <- stats::sd(x); if (sigma <= 0) sigma <- 1
  mu2 <- 1; R_K <- 1 / (2 * sqrt(pi))
  R_fpp <- 3 / (8 * sqrt(pi) * sigma^5)
  if (is.null(h)) h <- .morie_silverman_h(x)
  bias_part <- (h^4 / 4) * mu2^2 * R_fpp
  var_part  <- R_K / (n * h)
  mise <- bias_part + var_part
  h_opt <- (R_K / (n * mu2^2 * R_fpp))^(1/5)
  list(estimate = mise, bias_part = bias_part, var_part = var_part,
       h = h, h_opt = h_opt, R_fpp = R_fpp, sigma = sigma, n = n,
       method = "Fauzi MISE decomposition (Ch 1)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500); r <- fzmis(x)
# stopifnot(r$bias_part > 0 && r$var_part > 0)

#' @rdname fzmis
#' @keywords internal
#' @export
fauzi_mise_computation <- fzmis
