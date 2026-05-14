# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Bias-reduced KDFE via geometric extrapolation (Ch 2)
#'
#' Richardson-style extrapolation cancels the O(h^2) bias of the KDFE:
#' \deqn{\hat F_{br}(t) = (c^2 \hat F_h(t) - \hat F_{ch}(t))/(c^2-1).}
#'
#' @param x Numeric vector.
#' @param t Evaluation point; default median(x).
#' @param h Bandwidth; default = Silverman's rule.
#' @param c Extrapolation ratio (>1); default 2.
#' @return Named list with estimate, F_h, F_ch, se, h, c, t, n, method.
#' @importFrom stats median pnorm
#' @export
fzbrd <- function(x, t = NULL, h = NULL, c = 2) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                           method = "fzbrd - too few obs"))
  if (is.null(t)) t <- stats::median(x)
  if (is.null(h)) h <- .morie_silverman_h(x)
  if (c <= 1) stop("c must be > 1")
  F_h  <- mean(stats::pnorm((t - x) / h))
  F_ch <- mean(stats::pnorm((t - x) / (c * h)))
  F_br <- (c^2 * F_h - F_ch) / (c^2 - 1)
  var_F <- F_h * (1 - F_h) / n
  var_inflate <- (c^4 + 1) / (c^2 - 1)^2
  list(estimate = F_br, F_h = F_h, F_ch = F_ch,
       se = sqrt(var_F * var_inflate), h = h, c = c, t = t, n = n,
       method = "Fauzi bias-reduced KDFE (Ch 2)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500)
# r <- fzbrd(x, t = 0); stopifnot(abs(r$estimate - 0.5) < 0.1)

#' @rdname fzbrd
#' @keywords internal
#' @export
fauzi_bias_reduced_kdfe <- fzbrd
