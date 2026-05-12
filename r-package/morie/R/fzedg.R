# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: Edgeworth expansion for kernel quantile (Ch 3)
#'
#' One-term Edgeworth correction to the normal approximation of the
#' studentised kernel quantile.  Skewness of the indicator score
#' \eqn{\gamma_1=(1-2p)/\sqrt{p(1-p)}} and
#' \deqn{P(T_n \le z) \approx \Phi(z) - (\gamma_1/6)(z^2-1)\phi(z)/\sqrt n.}
#'
#' @param x Numeric vector (only n is used; result is asymptotic).
#' @param z Standardised critical value; default 1.96.
#' @param p Quantile probability; default 0.5.
#' @return Named list with estimate, normal_approx, edgeworth_correction,
#'   cornish_fisher_correction, skew, p1z, z, p, n, method.
#' @importFrom stats dnorm pnorm
#' @export
fzedg <- function(x, z = 1.96, p = 0.5) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(estimate = NA_real_, n = n,
                           method = "fzedg - too few obs"))
  skew <- (1 - 2 * p) / sqrt(p * (1 - p))
  p1z <- -(skew / 6) * (z^2 - 1)
  phi_z <- stats::dnorm(z); Phi_z <- stats::pnorm(z)
  correction <- p1z * phi_z / sqrt(n)
  cf_correction <- (skew / 6) * (z^2 - 1) / sqrt(n)
  list(estimate = Phi_z + correction,
       normal_approx = Phi_z,
       edgeworth_correction = correction,
       cornish_fisher_correction = cf_correction,
       skew = skew, p1z = p1z, z = z, p = p, n = n,
       method = "Fauzi Edgeworth expansion for kernel quantile (Ch 3)")
}

# CANONICAL TEST
# r <- fzedg(1:10, z = 1.96, p = 0.5); stopifnot(abs(r$skew) < 1e-12)

#' @rdname fzedg
#' @keywords internal
#' @export
fauzi_edgeworth_quantile <- fzedg
