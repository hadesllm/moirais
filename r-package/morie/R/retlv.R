# SPDX-License-Identifier: GPL-2.0-only
#' GEV return-level estimation (Coles 2001, Ch. 3)
#'
#' For block maxima with GEV(mu, sigma, xi), the T-block return level
#' is z_T = mu - (sigma/xi) (1 - (-log(1 - 1/T))^(-xi))  if xi != 0,
#' else z_T = mu - sigma log(-log(1 - 1/T)).  Delta-method SE uses the
#' diagonal of the observed information returned by \code{extvm}.
#'
#' @param x numeric vector of block maxima.
#' @param return_period numeric T (default 100).
#' @return list: z, se, return_period, mu, sigma, xi, n, method.
#' @keywords internal
retlv <- function(x, return_period = 100) {
  fit <- extvm(x)
  if (!is.finite(fit$mu %||% NA_real_))
    return(list(estimate = NA_real_, method = "Return level (GEV fit failed)"))
  mu <- fit$mu; sigma <- fit$sigma; xi <- fit$xi
  p <- 1 / return_period
  yp <- -log(1 - p)
  if (abs(xi) < 1e-6) {
    z <- mu - sigma * log(yp)
    d_mu <- 1; d_sig <- -log(yp); d_xi <- 0.5 * sigma * log(yp)^2
  } else {
    z <- mu - (sigma / xi) * (1 - yp^(-xi))
    d_mu <- 1
    d_sig <- -(1 / xi) * (1 - yp^(-xi))
    d_xi <- (sigma / xi^2) * (1 - yp^(-xi)) -
            (sigma / xi) * yp^(-xi) * log(yp)
  }
  var_z <- (d_mu * fit$se_mu)^2 + (d_sig * fit$se_sigma)^2 +
           (d_xi * fit$se_xi)^2
  se <- sqrt(max(0, var_z))
  list(z = as.numeric(z), estimate = as.numeric(z), se = as.numeric(se),
       return_period = as.numeric(return_period),
       mu = mu, sigma = sigma, xi = xi,
       n = length(x),
       method = "Return level (Coles 2001)")
}

# small null-coalesce helper local to this file
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# CANONICAL TEST
# set.seed(0); x <- evd::rgumbel(500, loc = 10, scale = 2)
# r <- retlv(x, return_period = 100)
# # 100-yr Gumbel level: 10 - 2*log(-log(0.99)) ~= 19.2
# stopifnot(r$z > 16, r$z < 23)

#' @rdname retlv
#' @keywords internal
#' @export
return_level <- retlv
