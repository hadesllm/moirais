# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: Boundary-free MRL via log-bijection (Ch 4)
#'
#' Avoids boundary bias of the kernel MRL near t=0 by smoothing on
#' the log-transformed scale.
#'
#' @param x Numeric vector, strictly positive.
#' @param t Evaluation point > 0; default = median(x).
#' @param h Bandwidth on log scale; default = Silverman.
#' @return Named list: estimate, se, S_hat, t, h, n, method.
#' @importFrom stats median pnorm
#' @export
fzmrb <- function(x, t = NULL, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (any(x <= 0)) stop("fzmrb requires strictly positive x")
  if (n < 2L) return(list(estimate = NA_real_, se = NA_real_, n = n,
                           method = "fzmrb - too few obs"))
  if (is.null(t)) t <- stats::median(x)
  if (t <= 0) stop("t must be positive")
  y <- log(x)
  if (is.null(h)) h <- .morie_silverman_h(y)
  s <- log(t)
  S_y <- mean(1 - stats::pnorm((s - y) / h))
  if (S_y <= 0) return(list(estimate = NA_real_, S_hat = S_y, n = n, t = t,
                             method = "fzmrb - S(t)=0"))
  d <- x - t; above <- d > 0
  if (!any(above)) return(list(estimate = 0, S_hat = S_y, n = n, t = t,
                                method = "fzmrb - no x>t"))
  m_hat <- mean(d[above])
  second <- mean((d[above])^2)
  sigma2 <- max((second - m_hat^2) / S_y, 0)
  list(estimate = m_hat, se = sqrt(sigma2 / n),
       S_hat = S_y, t = t, h = h, n = n,
       method = "Fauzi boundary-free MRL via log-bijection (Ch 4)")
}

# CANONICAL TEST
# set.seed(0); x <- rexp(2000, 1)
# r <- fzmrb(x, t = 0.5); stopifnot(abs(r$estimate - 1) < 0.2)

#' @rdname fzmrb
#' @keywords internal
#' @export
fauzi_mrl_boundary_free <- fzmrb
