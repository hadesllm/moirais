# SPDX-License-Identifier: AGPL-3.0-or-later

#' Fauzi: Kernel mean residual life asymptotics (Ch 4)
#'
#' Kernel-smoothed MRL \eqn{m(t)=E[X-t|X>t]} with Yang (1978)
#' asymptotic SE.
#'
#' @param x Numeric vector (lifetimes).
#' @param t Evaluation point; default = median(x).
#' @param h Bandwidth; default = Silverman's rule.
#' @return Named list with estimate, se, S_hat, t, h, n, method.
#' @importFrom stats median pnorm
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
fzmrl <- function(x, t = NULL, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, se = NA_real_, n = n,
                           method = "fzmrl - too few obs"))
  if (is.null(t)) t <- stats::median(x)
  if (is.null(h)) h <- .morie_silverman_h(x)
  S_t <- mean(1 - stats::pnorm((t - x) / h))
  if (S_t <= 0) return(list(estimate = NA_real_, se = NA_real_,
                             S_hat = S_t, n = n, t = t,
                             method = "fzmrl - S(t)=0"))
  d <- x - t; above <- d > 0
  if (!any(above)) return(list(estimate = 0, se = NA_real_,
                                 S_hat = S_t, n = n, t = t,
                                 method = "fzmrl - no x>t"))
  m_hat <- mean(d[above])
  second <- mean((d[above])^2)
  sigma2 <- max((second - m_hat^2) / S_t, 0)
  list(estimate = m_hat, se = sqrt(sigma2 / n),
       S_hat = S_t, t = t, h = h, n = n,
       method = "Fauzi kernel MRL asymptotic (Ch 4)")
}

# CANONICAL TEST
# set.seed(0); x <- rexp(2000, 1)
# r <- fzmrl(x, t = 0); stopifnot(abs(r$estimate - 1) < 0.15)

#' @rdname fzmrl
#' @keywords internal
#' @export
fauzi_mrl_asymptotic <- fzmrl
