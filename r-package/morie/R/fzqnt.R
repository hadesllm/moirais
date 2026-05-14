# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Asymptotic distribution of kernel quantile (Ch 3)
#'
#' \eqn{\sqrt n(\hat Q-Q) \to N(0,p(1-p)/f(Q)^2)}.
#'
#' @param x Numeric vector.
#' @param p Quantile probability in (0,1); default 0.5.
#' @param h Bandwidth; default = Silverman's rule.
#' @return Named list with estimate, se, p, h, density_at_Q, n, method.
#' @importFrom stats sd quantile dnorm pnorm uniroot
#' @export
fzqnt <- function(x, p = 0.5, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(estimate = NA_real_, se = NA_real_, n = n,
                           method = "fzqnt - too few obs"))
  if (p <= 0 || p >= 1) stop("p must be in (0,1)")
  if (is.null(h)) h <- .morie_silverman_h(x)
  Fhat <- function(t) mean(stats::pnorm((t - x) / h))
  lo <- min(x) - 5 * h - 1e-9; hi <- max(x) + 5 * h + 1e-9
  q_hat <- tryCatch(
    stats::uniroot(function(t) Fhat(t) - p, c(lo, hi))$root,
    error = function(e) as.numeric(stats::quantile(x, p, names = FALSE))
  )
  f_q <- mean(stats::dnorm((q_hat - x) / h) / h)
  se <- if (f_q > 0) sqrt(p * (1 - p) / n) / f_q else NA_real_
  list(estimate = q_hat, se = se, p = p, h = h,
       density_at_Q = f_q, n = n,
       method = "Fauzi kernel quantile asymptotic (Ch 3)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(1000)
# r <- fzqnt(x, p = 0.5); stopifnot(abs(r$estimate) < 0.15)

#' @rdname fzqnt
#' @keywords internal
#' @export
fauzi_kernel_quantile_asymptotic <- fzqnt
